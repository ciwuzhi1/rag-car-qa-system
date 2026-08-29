import os

# 列出项目文件
print("项目文件列表:")
for root, dirs, files in os.walk('.'):
    for file in files:
        file_path = os.path.join(root, file)
        print(f"{file_path} - 大小: {os.path.getsize(file_path)} 字节")

# 检查特定文件
files_to_check = [
    'app.py',
    'templates/index.html',
    'templates/admin.html',
    'templates/knowledge.html',
    'templates/upload_knowledge.html',
    'templates/faq.html',
    'templates/feedback.html',
    'static/css/style.css'
]

print("\n文件状态检查:")
for file_path in files_to_check:
    if os.path.exists(file_path):
        print(f"{file_path} - 存在，大小: {os.path.getsize(file_path)} 字节")
    else:
        print(f"{file_path} - 不存在") 