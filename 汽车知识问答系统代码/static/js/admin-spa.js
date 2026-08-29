/* 管理后台无刷新切换：拦截管理页之间的链接，fetch 拉取页面后替换 body 并重新执行脚本，
   配合透明度过渡实现丝滑切换，彻底避免整页重载的闪屏。 */
(function () {
    // 只接管这四个管理页之间的互相切换，其他链接（首页、登出等）仍走整页跳转
    var PAGES = ['/admin', '/knowledge', '/upload_knowledge', '/api_keys'];

    // 事件委托绑定在 document 上，body 被替换后依然有效
    document.addEventListener('click', function (e) {
        if (e.defaultPrevented) return;
        var a = e.target.closest('a');
        if (!a) return;
        var url = a.getAttribute('href');
        if (!url || PAGES.indexOf(url) === -1) return;
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        navigateTo(url, true);
    });

    window.addEventListener('popstate', function () {
        if (PAGES.indexOf(location.pathname) !== -1) navigateTo(location.pathname, false);
    });

    function navigateTo(url, push) {
        var container = document.querySelector('.container-fluid');
        if (container) {
            container.style.transition = 'opacity 0.15s ease';
            container.style.opacity = '0.35';
        }

        fetch(url, { headers: { 'X-Requested-With': 'SPA' } })
            .then(function (r) {
                if (!r.ok) throw new Error(r.status);
                return r.text();
            })
            .then(function (html) {
                var doc = new DOMParser().parseFromString(html, 'text/html');
                // 目标必须是管理页（登录过期时会返回登录页，此时整页跳转兜底）
                if (!doc.querySelector('.navbar-app')) {
                    window.location.href = url;
                    return;
                }
                document.body.innerHTML = doc.body.innerHTML;
                document.title = doc.title;
                executeScripts(doc);
                updateActive(url);
                if (push) history.pushState({ spa: true }, '', url);
                window.scrollTo(0, 0);

                var c = document.querySelector('.container-fluid');
                if (c) {
                    c.style.opacity = '0';
                    requestAnimationFrame(function () {
                        c.style.transition = 'opacity 0.25s ease';
                        c.style.opacity = '1';
                    });
                }
            })
            .catch(function () {
                window.location.href = url;  // 网络异常等情况下整页跳转兜底
            });
    }

    // 重新执行目标页的内联脚本；外部脚本（Bootstrap 等）已加载过，跳过
    function executeScripts(doc) {
        doc.querySelectorAll('body script').forEach(function (old) {
            if (old.src) return;
            var s = document.createElement('script');
            // 包一层 IIFE：避免多次切换后 const/let/function 全局重复声明报错
            s.textContent = '(function () {\n' + old.textContent + '\n})();';
            document.body.appendChild(s);
            s.remove();
        });
    }

    function updateActive(url) {
        document.querySelectorAll('.list-group a').forEach(function (a) {
            a.classList.toggle('active', a.getAttribute('href') === url);
        });
    }
})();
