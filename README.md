# 技术文章网站框架

这是一个适合部署到 GitHub Pages 的轻量静态技术文章网站。当前版本不依赖构建工具，直接打开 `index.html` 或推送到 `username.github.io` 仓库即可访问。

## 目录结构

```text
.
├── index.html
├── posts
│   └── technical-note-template.html
├── assets
│   ├── css
│   │   └── styles.css
│   ├── img
│   │   └── favicon.svg
│   └── js
│       └── main.js
└── README.md
```

## 写新文章

1. 复制 `posts/technical-note-template.html`，改成新的英文文件名，例如 `posts/my-first-note.html`。
2. 修改新文件里的标题、日期、标签和正文。
3. 在 `index.html` 的“最新文章”和“归档”里添加链接。
4. 提交并推送到 GitHub，GitHub Pages 会自动更新。

## 部署到 GitHub Pages

1. 仓库名使用 `你的GitHub用户名.github.io`。
2. 将本目录内容提交并推送到仓库默认分支。
3. 打开仓库的 `Settings -> Pages`，确认 Source 选择默认分支根目录。
4. 等待 GitHub Pages 构建完成后，访问 `https://你的GitHub用户名.github.io`。
