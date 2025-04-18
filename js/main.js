/**
 * 主JavaScript文件
 */

document.addEventListener('DOMContentLoaded', function() {
    // 设置当前页面在导航栏中的活跃状态
    setActiveNavItem();
    
    // 设置滚动到顶部按钮
    setupScrollToTop();
    
    // 设置主题切换功能（如果未来实现）
    // setupThemeToggle();
    
    // 设置搜索功能（如果未来实现）
    // setupSearch();
});

/**
 * 设置当前页面在导航栏中的活跃状态
 */
function setActiveNavItem() {
    const currentPage = window.location.pathname.split('/').pop();
    
    // 默认为首页
    let activeLink = 'index.html';
    
    if (currentPage) {
        activeLink = currentPage;
    }
    
    // 找到对应的导航项并设置为活跃
    const navLinks = document.querySelectorAll('nav a');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === activeLink) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * 设置滚动到顶部按钮
 */
function setupScrollToTop() {
    // 创建滚动到顶部按钮
    const scrollTopBtn = document.createElement('button');
    scrollTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    scrollTopBtn.className = 'scroll-top-btn';
    scrollTopBtn.style.display = 'none';
    scrollTopBtn.style.position = 'fixed';
    scrollTopBtn.style.bottom = '20px';
    scrollTopBtn.style.right = '20px';
    scrollTopBtn.style.zIndex = '99';
    scrollTopBtn.style.border = 'none';
    scrollTopBtn.style.outline = 'none';
    scrollTopBtn.style.backgroundColor = 'var(--primary-color)';
    scrollTopBtn.style.color = 'white';
    scrollTopBtn.style.cursor = 'pointer';
    scrollTopBtn.style.padding = '10px 15px';
    scrollTopBtn.style.borderRadius = '50%';
    scrollTopBtn.style.fontSize = '16px';
    scrollTopBtn.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.3)';
    scrollTopBtn.style.transition = 'all 0.3s ease';
    
    document.body.appendChild(scrollTopBtn);
    
    // 监听滚动事件
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            scrollTopBtn.style.display = 'block';
        } else {
            scrollTopBtn.style.display = 'none';
        }
    });
    
    // 点击事件
    scrollTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

/**
 * 格式化日期
 * @param {Date} date - 日期对象
 * @returns {string} - 格式化后的日期字符串 (YYYY-MM-DD)
 */
function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * 计算文章阅读时间
 * @param {string} content - 文章内容
 * @returns {number} - 预计阅读时间（分钟）
 */
function calculateReadTime(content) {
    // 假设平均阅读速度为每分钟250个字
    const wordsPerMinute = 250;
    const words = content.trim().split(/\s+/).length;
    return Math.ceil(words / wordsPerMinute);
}

/**
 * 页面动画和交互效果
 */
function animateOnScroll() {
    // 使用IntersectionObserver API检测元素是否在视口中
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1
    });

    // 获取所有需要动画的元素
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
} 