# 个人网站框架

这是一个适合部署到 GitHub Pages 的轻量静态个人网站骨架。当前版本不依赖构建工具，直接打开 `index.html` 或推送到 `username.github.io` 仓库即可访问。

## 目录结构

```text
.
├── index.html
├── assets
│   ├── css
│   │   └── styles.css
│   ├── img
│   │   └── profile-placeholder.svg
│   └── js
│       └── main.js
└── README.md
```

## 下一步替换内容

1. 把 `index.html` 里的 `你的名字`、简介、项目、文章和联系方式替换成真实信息。
2. 将 `assets/img/profile-placeholder.svg` 替换成你的头像、照片或个人标识。
3. 把项目卡片和文章链接中的 `href="#"` 改为真实链接。
4. 如果你使用自定义域名，后续可以在根目录新增 `CNAME` 文件。

## 部署到 GitHub Pages

1. 新建仓库，仓库名使用 `你的GitHub用户名.github.io`。
2. 将本目录内容提交并推送到仓库默认分支。
3. 打开仓库的 `Settings -> Pages`，确认 Source 选择默认分支根目录。
4. 等待 GitHub Pages 构建完成后，访问 `https://你的GitHub用户名.github.io`。
