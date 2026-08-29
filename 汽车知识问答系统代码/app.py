from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, session, send_file, flash, g)
import os
import json
import uuid
import datetime
import functools
import requests
import csv
import re
import io
import sqlite3

app = Flask(__name__)

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# 会话密钥：首次运行时生成并持久化，保证重启后已登录的 session 不失效
SECRET_FILE = os.path.join(DATA_DIR, '.secret_key')
if os.path.exists(SECRET_FILE):
    with open(SECRET_FILE, 'rb') as f:
        app.secret_key = f.read()
else:
    app.secret_key = os.urandom(24)
    with open(SECRET_FILE, 'wb') as f:
        f.write(app.secret_key)

# 数据库文件（SQLite）
DB_FILE = os.path.join(DATA_DIR, 'knowledge_qa.db')

# 旧版 JSON 数据文件（仅用于首次启动时迁移到 SQLite）
KNOWLEDGE_DB_FILE = os.path.join(DATA_DIR, 'knowledge_base.json')
QUESTIONS_FILE = os.path.join(DATA_DIR, 'questions.json')
FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.json')
IMPORT_LOGS_FILE = os.path.join(DATA_DIR, 'import_logs.json')

# 管理员账号（可通过环境变量覆盖）
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# 大模型API配置（OpenAI兼容格式，均可通过环境变量覆盖）
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_API_URL = os.environ.get('OPENAI_API_URL', 'https://api.openai.com/v1/chat/completions')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
# 智谱 GLM 大模型（OpenAI 兼容接口，配置 ZHIPU_API_KEY 即可使用）
ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY', '')
ZHIPU_API_URL = os.environ.get('ZHIPU_API_URL',
                               'https://open.bigmodel.cn/api/paas/v4/chat/completions')
ZHIPU_MODEL = os.environ.get('ZHIPU_MODEL', 'glm-4.5-air')


# ============ 数据库 ============

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            keywords TEXT DEFAULT '',
            source TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            time TEXT,
            model TEXT,
            keywords TEXT,
            type TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_type TEXT,
            title TEXT,
            content TEXT,
            contact TEXT,
            time TEXT
        );
        CREATE TABLE IF NOT EXISTS import_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            message TEXT,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    ''')
    # 旧 JSON 数据一次性迁移（仅在对应表为空时导入）
    _migrate_json(conn, KNOWLEDGE_DB_FILE, 'knowledge',
                  ['question', 'answer', 'keywords'])
    _migrate_json(conn, QUESTIONS_FILE, 'questions',
                  ['id', 'text', 'time', 'model', 'keywords', 'type'])
    _migrate_json(conn, FEEDBACK_FILE, 'feedback',
                  ['feedback_type', 'title', 'content', 'contact', 'time'])
    _migrate_json(conn, IMPORT_LOGS_FILE, 'import_logs',
                  ['time', 'message', 'status'])
    conn.commit()
    conn.close()


def _migrate_json(conn, json_file, table, columns):
    if not os.path.exists(json_file):
        return
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            rows = json.load(f)
    except (ValueError, OSError) as e:
        print(f"迁移 {json_file} 失败: {e}")
        return
    if not isinstance(rows, list) or not rows:
        return
    if conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] > 0:
        return
    cols = ','.join(columns)
    marks = ','.join('?' * len(columns))
    conn.executemany(
        f'INSERT INTO {table} ({cols}) VALUES ({marks})',
        [tuple(str(r.get(c, '')) for c in columns) for r in rows
         if isinstance(r, dict)])
    print(f"已从 {json_file} 迁移 {len(rows)} 条数据到 {table} 表")


# ============ 系统设置（密钥等，存 SQLite） ============

# 可在界面配置的密钥项：provider 名 -> 环境变量兜底值
API_KEY_PROVIDERS = {
    'zhipu': ('ZHIPU_API_KEY', '智谱GLM'),
    'openai': ('OPENAI_API_KEY', 'OpenAI'),
}


def get_setting(key, default=''):
    row = get_db().execute('SELECT value FROM settings WHERE key = ?',
                           (key,)).fetchone()
    return row['value'] if row else default


def set_setting(key, value):
    db = get_db()
    db.execute('INSERT INTO settings (key, value) VALUES (?, ?) '
               'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
               (key, value))
    db.commit()


def get_api_key(provider):
    """读取密钥：界面配置优先，环境变量兜底"""
    env_name = API_KEY_PROVIDERS.get(provider, ('', ''))[0]
    return get_setting(f'api_key_{provider}', '') or os.environ.get(env_name, '')


# ============ 登录鉴权 ============

def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if session.get('logged_in'):
            return view(*args, **kwargs)
        # 浏览器访问页面时跳转登录页，fetch/ajax 调用时返回 401 JSON
        if request.method == 'GET' and 'text/html' in request.headers.get('Accept', ''):
            return redirect(url_for('login'))
        return jsonify({'status': 'error', 'message': '请先登录管理后台'}), 401
    return wrapped


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('admin'))
        error = '用户名或密码错误！'
    if session.get('logged_in'):
        return redirect(url_for('admin'))
    return render_template('login.html', error=error)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))


# ============ API密钥配置页面 ============

@app.route('/api_keys')
@login_required
def api_keys_page():
    """密钥配置页面"""
    configured = {p: bool(get_api_key(p)) for p in API_KEY_PROVIDERS}
    masked = {}
    for p in API_KEY_PROVIDERS:
        key = get_api_key(p)
        # 已配置的密钥只回显前4位与后4位
        masked[p] = f"{key[:4]}****{key[-4:]}" if len(key) > 8 else ''
    return render_template('api_keys.html', configured=configured, masked=masked,
                           zhipu_model=ZHIPU_MODEL, openai_model=OPENAI_MODEL)


@app.route('/save_api_key', methods=['POST'])
@login_required
def save_api_key():
    data = request.get_json(silent=True) or {}
    provider = data.get('provider', '')
    api_key = (data.get('api_key') or '').strip()
    if provider not in API_KEY_PROVIDERS:
        return jsonify({'status': 'error', 'message': '不支持的服务商'})
    if not api_key:
        # 传空表示清除该密钥
        set_setting(f'api_key_{provider}', '')
        return jsonify({'status': 'success', 'message': '密钥已清除'})
    set_setting(f'api_key_{provider}', api_key)
    return jsonify({'status': 'success',
                    'message': f"{API_KEY_PROVIDERS[provider][1]}密钥已保存，现在可以直接使用了"})


# ============ 页面路由 ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin')
@login_required
def admin():
    db = get_db()
    questions = [dict(r) for r in
                 db.execute('SELECT * FROM questions ORDER BY time DESC')]
    return render_template('admin.html', questions=questions)


@app.route('/knowledge')
@login_required
def knowledge_manager():
    db = get_db()
    search_term = request.args.get('search', '')
    if search_term:
        like = f'%{search_term}%'
        rows = db.execute(
            "SELECT * FROM knowledge WHERE question LIKE ? OR answer LIKE ? "
            "OR keywords LIKE ? ORDER BY id DESC", (like, like, like))
    else:
        rows = db.execute('SELECT * FROM knowledge ORDER BY id DESC')
    knowledge = [dict(r) for r in rows]
    return render_template('knowledge.html', knowledge_base=knowledge)


@app.route('/upload_knowledge')
@login_required
def upload_knowledge():
    db = get_db()
    logs = [dict(r) for r in
            db.execute('SELECT * FROM import_logs ORDER BY id DESC LIMIT 50')]
    return render_template('upload_knowledge.html', logs=logs)


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/feedback')
def feedback():
    success = request.args.get('success', '')
    error = request.args.get('error', '')
    return render_template('feedback.html', success=success, error=error)


# ============ 问答 API ============

@app.route('/api_status')
def api_status():
    db = get_db()
    kb_size = db.execute('SELECT COUNT(*) FROM knowledge').fetchone()[0]
    return jsonify({
        'status': 'ok',
        'knowledge_base_size': kb_size,
        'current_model': session.get('model', 'offline')
    })


@app.route('/check_model', methods=['POST'])
def check_model():
    model_type = (request.get_json(silent=True) or {}).get('model', 'offline')
    if model_type == 'offline':
        return jsonify({'status': 'available'})
    if model_type == 'openai':
        api_key, api_url = get_api_key('openai'), OPENAI_API_URL
    elif model_type == 'zhipu':
        api_key, api_url = get_api_key('zhipu'), ZHIPU_API_URL
    else:
        return jsonify({'status': 'unavailable'})
    if not api_key:
        return jsonify({'status': 'unavailable'})
    # 通过 OpenAI 兼容的 /models 接口探测连通性
    models_url = api_url.rsplit('/chat/completions', 1)[0] + '/models'
    try:
        resp = requests.get(models_url,
                            headers={'Authorization': f'Bearer {api_key}'},
                            timeout=5)
        status = 'available' if resp.status_code == 200 else 'unavailable'
    except requests.RequestException:
        status = 'unavailable'
    return jsonify({'status': status})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    model_type = data.get('model', 'offline')

    if not question:
        return jsonify({'answer': '请输入问题'})

    # 记录问题
    question_record = (
        str(uuid.uuid4()),
        question,
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        model_type,
        extract_keywords(question),
        'user_question'
    )
    db = get_db()
    db.execute('INSERT INTO questions (id, text, time, model, keywords, type) '
               'VALUES (?, ?, ?, ?, ?, ?)', question_record)
    db.commit()

    answer, model_used = get_answer(question, model_type)
    return jsonify({'answer': answer, 'model_used': model_used})


def extract_keywords(text):
    # 中文按单字/短语切分效果有限，这里按标点与空白切词，保留长度>1的片段
    words = re.split(r'[\s，。？！、；：,.?!;:]+', text)
    return ','.join([w for w in words if len(w) > 1])


# 从内置知识库获取回答（简单模式）
def get_offline_answer(question):
    default_answers = [
        "汽车应定期进行保养，包括更换机油、机滤、空滤、汽滤等，一般每5000-7500公里进行一次小保养。",
        "发动机是汽车的动力源，内燃机通过燃烧燃料推动活塞运动，从而产生动力。",
        "电动汽车使用电力驱动，无排放污染；燃油车使用内燃机，有尾气排放但续航更长。",
        "正确的轮胎气压可以在车门边框、油箱盖内侧或车主手册中找到，一般轿车冷胎时为220-240kPa。",
        "汽车电池没电时，可以使用电瓶搭线方法，即使用另一辆车的电瓶和搭线电缆帮助启动。"
    ]

    if "保养" in question or "维护" in question:
        return default_answers[0]
    elif "发动机" in question or "原理" in question:
        return default_answers[1]
    elif "电动" in question or "燃油" in question or "区别" in question:
        return default_answers[2]
    elif "轮胎" in question or "气压" in question:
        return default_answers[3]
    elif "电池" in question or "搭线" in question:
        return default_answers[4]

    return "作为汽车知识助手，我目前的内置知识库中没有这个问题的确切答案。您可以尝试使用智谱GLM或OpenAI模式获取更详细的回答，或者通过用户反馈页面提交这个问题。"


RAG_SIM_THRESHOLD = 0.55   # 语义相似度低于该值时认为知识库没有命中


def rag_search(rows, question, top_k=3):
    """用本地 bge-m3 嵌入模型做语义检索；不可用时返回 None（调用方回退关键词匹配）"""
    if not rows:
        return []
    try:
        import rag
        if not rag.is_available():
            return None
        return rag.search(rows, question, top_k=top_k)
    except Exception as e:
        print(f'RAG 检索不可用，回退关键词匹配: {e}')
        return None


def keyword_match(db, question):
    """原有匹配逻辑：精确 → 双向包含 → 关键词命中"""
    row = db.execute(
        'SELECT answer FROM knowledge WHERE LOWER(question) = LOWER(?)',
        (question,)).fetchone()
    if not row:
        row = db.execute(
            'SELECT answer FROM knowledge WHERE INSTR(LOWER(?), LOWER(question)) > 0 '
            'OR INSTR(LOWER(question), LOWER(?)) > 0',
            (question, question)).fetchone()
    if not row:
        # 关键词匹配：把每条知识的 keywords 拆开逐个匹配
        for r in db.execute("SELECT answer, keywords FROM knowledge "
                            "WHERE keywords IS NOT NULL AND keywords != ''"):
            for kw in re.split(r'[,，、;；\s]+', r['keywords']):
                if len(kw) > 1 and kw.lower() in question.lower():
                    return r['answer']
    return row['answer'] if row else None


def get_answer(question, model_type):
    """返回 (回答, 来源标识)；来源标识用于前端展示该回答由谁生成"""
    session['model'] = model_type
    db = get_db()
    rows = [dict(r) for r in db.execute('SELECT * FROM knowledge')]

    # 1) RAG 语义检索优先；模型不可用时回退关键词匹配
    hits = rag_search(rows, question)
    if hits:
        best_row, best_score = hits[0]
        if best_score >= RAG_SIM_THRESHOLD:
            return best_row['answer'], '知识库RAG · 相似度%.2f' % best_score
    else:
        answer = keyword_match(db, question)
        if answer:
            return answer, '知识库关键词匹配'

    # 2) 未命中知识库
    if model_type == 'offline':
        return get_offline_answer(question), '内置回答'
    elif model_type in ('openai', 'zhipu'):
        # 简单 RAG：把检索到的相关知识作为上下文注入大模型
        context = ''
        if hits:
            refs = [f"【{r['question']}】{r['answer']}"
                    for r, s in hits if s >= 0.30]
            if refs:
                context = '\n'.join(refs)
        return call_llm(model_type, question, context)
    else:
        return "不支持的模型类型，请选择内置回答、智谱GLM或OpenAI。", None


def call_llm(provider, question, context=''):
    """调用 OpenAI 兼容的 chat/completions 接口，返回 (回答, 实际使用的模型名)"""
    if provider == 'openai':
        api_key, api_url, model_name, env_name = \
            get_api_key('openai'), OPENAI_API_URL, OPENAI_MODEL, 'OPENAI_API_KEY'
        provider_label = 'OpenAI'
    elif provider == 'zhipu':
        api_key, api_url, model_name, env_name = \
            get_api_key('zhipu'), ZHIPU_API_URL, ZHIPU_MODEL, 'ZHIPU_API_KEY'
        provider_label = '智谱GLM'
    else:
        raise ValueError(f'未知的大模型服务商: {provider}')

    if not api_key:
        return (f"{provider_label}API尚未配置密钥，请在管理后台“API密钥配置”页面填写，"
                f"或使用内置回答模式。"), None

    system_prompt = ('你是一位专业的汽车知识助手，请用中文简洁、准确地回答汽车相关的'
                     '问题。如果问题与汽车无关，请礼貌地引导用户询问汽车知识。')
    if context:
        system_prompt += ('\n\n以下是知识库中检索到的相关资料，请优先依据它们回答，'
                          '资料没有覆盖的部分可以补充通用知识：\n' + context)
    try:
        resp = requests.post(
            api_url,
            headers={'Authorization': f'Bearer {api_key}'},
            json={
                'model': model_name,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': question}
                ],
                'temperature': 0.7
            },
            timeout=30)
        resp.raise_for_status()
        data = resp.json()
        answer = data['choices'][0]['message']['content'].strip()
        # 优先展示服务端返回的实际模型名（可能与请求的 model 不同，如被网关改写）
        actual_model = data.get('model') or model_name
        return (answer or '大模型返回了空回答，请稍后再试。'), \
            f'{provider_label} · {actual_model}'
    except requests.exceptions.Timeout:
        return '调用大模型API超时，请稍后再试，或使用内置回答模式。', None
    except requests.exceptions.RequestException as e:
        return f'调用大模型API失败：{e}。请检查网络与API密钥配置，或使用内置回答模式。', None
    except (KeyError, IndexError, ValueError):
        return '大模型API返回了无法解析的内容，请稍后再试。', None


# ============ 知识库管理 API ============

@app.route('/add_knowledge', methods=['POST'])
@login_required
def add_knowledge():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    answer = (data.get('answer') or '').strip()
    keywords = (data.get('keywords') or '').strip()
    if not question or not answer:
        return jsonify({'status': 'error', 'message': '问题和答案不能为空'})
    db = get_db()
    db.execute('INSERT INTO knowledge (question, answer, keywords, source) '
               'VALUES (?, ?, ?, ?)', (question, answer, keywords, 'manual'))
    db.commit()
    return jsonify({'status': 'success'})


@app.route('/get_knowledge/<int:knowledge_id>')
@login_required
def get_knowledge(knowledge_id):
    row = get_db().execute('SELECT * FROM knowledge WHERE id = ?',
                           (knowledge_id,)).fetchone()
    if not row:
        return jsonify({'status': 'error', 'message': '知识条目不存在'}), 404
    return jsonify({k: row[k] for k in ('id', 'question', 'answer', 'keywords')})


@app.route('/update_knowledge/<int:knowledge_id>', methods=['POST'])
@login_required
def update_knowledge(knowledge_id):
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    answer = (data.get('answer') or '').strip()
    keywords = (data.get('keywords') or '').strip()
    if not question or not answer:
        return jsonify({'status': 'error', 'message': '问题和答案不能为空'})
    db = get_db()
    cur = db.execute('UPDATE knowledge SET question = ?, answer = ?, keywords = ? '
                     'WHERE id = ?', (question, answer, keywords, knowledge_id))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({'status': 'error', 'message': '知识条目不存在'}), 404
    return jsonify({'status': 'success'})


@app.route('/delete_knowledge/<int:knowledge_id>', methods=['POST'])
@login_required
def delete_knowledge(knowledge_id):
    db = get_db()
    cur = db.execute('DELETE FROM knowledge WHERE id = ?', (knowledge_id,))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({'status': 'error', 'message': '知识条目不存在'}), 404
    return jsonify({'status': 'success'})


@app.route('/clear_knowledge_base', methods=['POST'])
@login_required
def clear_knowledge_base():
    db = get_db()
    db.execute('DELETE FROM knowledge')
    db.commit()
    return jsonify({'status': 'success'})


@app.route('/export_knowledge')
@login_required
def export_knowledge():
    rows = [dict(r) for r in get_db().execute(
        'SELECT question, answer, keywords FROM knowledge ORDER BY id')]
    payload = json.dumps(rows, ensure_ascii=False, indent=2).encode('utf-8')
    buf = io.BytesIO(payload)
    return send_file(buf, as_attachment=True,
                     download_name='knowledge_base_export.json',
                     mimetype='application/json')


@app.route('/delete_question/<question_id>', methods=['POST'])
@login_required
def delete_question(question_id):
    db = get_db()
    cur = db.execute('DELETE FROM questions WHERE id = ?', (question_id,))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({'status': 'error', 'message': '问题记录不存在'}), 404
    return jsonify({'status': 'success'})


# ============ 知识导入 ============

def _parse_knowledge_json(text):
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError('JSON内容必须是数组格式')
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append({
                'question': str(item.get('question', '')).strip(),
                'answer': str(item.get('answer', '')).strip(),
                'keywords': str(item.get('keywords', '')).strip()
            })
    return result


def _parse_knowledge_txt(text):
    result = []
    blocks = [b for b in re.split(r'\n\s*\n', text) if b.strip()]
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        result.append({
            'question': lines[0],
            'answer': lines[1],
            'keywords': lines[2] if len(lines) > 2 else ''
        })
    return result


def _parse_knowledge_csv(text):
    result = []
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if rows and rows[0] and rows[0][0].strip() in ('问题', 'question'):
        rows = rows[1:]
    for row in rows:
        if not row or not row[0].strip():
            continue
        result.append({
            'question': row[0].strip(),
            'answer': row[1].strip() if len(row) > 1 else '',
            'keywords': ','.join(k.strip() for k in row[2:] if k.strip())
        })
    return result


def _parse_knowledge_md(text):
    result = []
    sections = re.split(r'^#{2,6}\s+', text, flags=re.MULTILINE)
    for section in sections[1:]:
        lines = section.splitlines()
        question = lines[0].strip()
        answer = '\n'.join(l for l in lines[1:]).strip()
        if question and answer:
            result.append({'question': question, 'answer': answer,
                           'keywords': ''})
    return result


KNOWLEDGE_PARSERS = {
    'json': _parse_knowledge_json,
    'txt': _parse_knowledge_txt,
    'csv': _parse_knowledge_csv,
    'md': _parse_knowledge_md,
}


def _save_knowledge_items(items, source):
    db = get_db()
    count = 0
    for item in items:
        if item.get('question') and item.get('answer'):
            db.execute('INSERT INTO knowledge (question, answer, keywords, source) '
                       'VALUES (?, ?, ?, ?)',
                       (item['question'], item['answer'],
                        item.get('keywords', ''), source))
            count += 1
    db.commit()
    return count


def _add_import_log(message, status):
    db = get_db()
    db.execute('INSERT INTO import_logs (time, message, status) VALUES (?, ?, ?)',
               (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                message, status))
    db.commit()


@app.route('/import_knowledge_text', methods=['POST'])
@login_required
def import_knowledge_text():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    fmt = data.get('format', 'json')
    parser = KNOWLEDGE_PARSERS.get(fmt)
    if not parser:
        return jsonify({'status': 'error', 'message': f'不支持的格式: {fmt}'})
    try:
        items = parser(text)
    except ValueError as e:
        _add_import_log(f'直接输入导入失败: {e}', 'error')
        return jsonify({'status': 'error', 'message': f'解析失败: {e}'})
    except Exception as e:
        _add_import_log(f'直接输入导入失败: {e}', 'error')
        return jsonify({'status': 'error', 'message': f'解析失败: {e}'})
    count = _save_knowledge_items(items, f'import_{fmt}')
    if count == 0:
        message = '未解析到有效的知识条目，请检查内容格式'
        _add_import_log(f'直接输入导入失败: {message}（{fmt} 格式）', 'error')
        return jsonify({'status': 'error', 'message': message})
    _add_import_log(f'通过直接输入导入了{count}条知识（{fmt} 格式）', 'success')
    return jsonify({'status': 'success', 'count': count})


@app.route('/upload_knowledge_file', methods=['POST'])
@login_required
def upload_knowledge_file():
    file = request.files.get('file')
    if not file or not file.filename:
        flash('请选择要上传的文件', 'error')
        return redirect(url_for('upload_knowledge'))

    ext = os.path.splitext(file.filename)[1].lstrip('.').lower()
    ext_map = {'json': 'json', 'txt': 'txt', 'csv': 'csv',
               'md': 'md', 'markdown': 'md'}
    fmt = ext_map.get(ext)
    if not fmt:
        flash('不支持的文件格式，请上传 JSON / TXT / CSV / Markdown 文件', 'error')
        return redirect(url_for('upload_knowledge'))

    text = file.read().decode('utf-8', errors='replace')
    try:
        items = KNOWLEDGE_PARSERS[fmt](text)
    except Exception as e:
        _add_import_log(f'文件 {file.filename} 导入失败: {e}', 'error')
        flash(f'导入失败，文件内容解析出错: {e}', 'error')
        return redirect(url_for('upload_knowledge'))

    count = _save_knowledge_items(items, 'import_file')
    if count == 0:
        message = '未解析到有效的知识条目，请检查文件内容格式'
        _add_import_log(f'文件 {file.filename} 导入失败: {message}', 'error')
        flash(message, 'error')
    else:
        _add_import_log(f'通过文件 {file.filename} 导入了{count}条知识'
                        f'（{fmt} 格式）', 'success')
        flash(f'导入成功，共导入{count}条知识', 'success')
    return redirect(url_for('upload_knowledge'))


# ============ 用户反馈 ============

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_type = request.form.get('feedback_type', '').strip()
    title = request.form.get('title', '').strip()
    content = request.form.get('content', '').strip()
    contact = request.form.get('contact', '').strip()

    if not feedback_type or not title or not content:
        return redirect(url_for('feedback',
                                error='请完整填写反馈类型、标题和详细内容'))

    db = get_db()
    db.execute('INSERT INTO feedback (feedback_type, title, content, contact, time) '
               'VALUES (?, ?, ?, ?, ?)',
               (feedback_type, title, content, contact,
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # 勾选了“添加到知识库”时，把反馈的问题与内容作为知识条目入库
    if request.form.get('add_to_knowledge'):
        db.execute('INSERT INTO knowledge (question, answer, keywords, source) '
                   'VALUES (?, ?, ?, ?)', (title, content, '', 'feedback'))
        db.commit()
        _add_import_log(f'来自用户反馈的知识已入库: {title}', 'success')
    else:
        db.commit()

    return redirect(url_for('feedback', success='感谢您的反馈，我们会尽快处理！'))


# 启动应用
init_db()

if __name__ == '__main__':
    app.run(debug=True)
