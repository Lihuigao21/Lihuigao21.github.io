# 歌曲展示固定模板

以已确认的《Photograph》页面为视觉基准：暖纸色整页构图、左侧唱片与歌曲信息、右侧无框歌词、横贯页面的真实频谱，下方保留随记、署名、时间微调与独立留言。手机使用同一套响应式布局。后续歌曲默认沿用这一模板。

`song-display.html` 固定结构，页面共用 `assets/css/music-visualizer.css` 和 `assets/js/music-visualizer.js`。生成后的 HTML 可独立发布，不需要构建服务或浏览器模板引擎。模板自身不加入生活记录索引和 sitemap。

将内容 JSON 放在本地忽略的 `articles/` 任务目录。以下是字段示意，音频路径和文字需要替换为实际素材：

```json
{
  "date": "2026-11-04",
  "song": "歌曲名称",
  "performer": "此演唱版本的演唱者",
  "media": "assets/media/song-name.mp3?v=1",
  "media_type": "audio/mpeg",
  "duration_seconds": 240,
  "lyrics_language": "zh-CN",
  "lyrics": ["第一段第一行\n第一段第二行", "第二段第一行"],
  "note": ["用户原文第一段", "用户原文第二段"],
  "credit": "原唱、演唱版本及素材署名"
}
```

`lyrics` 每项是一段，项内 `\n` 换行；不需要另填歌名段落。`note` 每项是一段原文，项内换行同样保留。所有内容按纯文本转义，不能注入 HTML。媒体必须是已存在的 `assets/media/` 或 `assets/data/` 下仓库相对路径；音频通常放在前者。先检查 MP4 的轨道，只有音频时使用 `audio/mp4`。

有与演唱版本匹配的真实时间轴时，可选增加：

```json
{
  "timings": "assets/data/song-name-timing.json",
  "timing_note": "该时间轴对应的版本及实际校准情况。",
  "timing_source_url": "https://example.org/actual-timing-source"
}
```

时间轴 JSON 为 `{"lineStarts": [12.3, 15.6, 15.6]}`：每个歌词行有一个非负起始秒数，按时间排序，共享时间不拆开。数量必须与全部歌词行数相等，且起点早于音频结束。时间来源 URL 可省略；时间轴未提供时，保留完整静态歌词，不显示跟随或校准控件，不自动生成时间。

从仓库根目录运行：

```text
python scripts/new-song-page.py --config articles/song-input.json --output life/art-song-name-2026-11-04.html
```

生成器拒绝覆盖现有文件，也不会修改来源素材、音频、索引或 sitemap。修改旧歌曲时直接维护其已有规范页面，保留已确认的标题、日期、置顶及媒体裁切。

发布新歌时：

1. 核对歌曲名、演唱版本、实际时长、媒体类型、署名及用户原文。保留完整静态歌词、原生音频控件和直属 `article` 的 `.post-nav`，后者用于挂载留言。
2. 将正式页面加入艺术评论的“记录索引”，更新 `life.html` 最新十篇（严格十篇，日期倒序）、首页生活记录日期及 `sitemap.xml`。保持已有置顶。
3. 在桌面、390px 与 320px 检查布局，实际播放、暂停和拖动进度；有时间轴时检查歌词跟随、点击跳转和相同时间的多行歌词。
4. 核查真实频谱、关闭动效后的暂停与恢复、减少动态效果和无 JavaScript / 时间轴加载失败时的原生播放及完整歌词。
5. 扫描公开内容的隐私与密钥，检查本地链接，提交、推送后比较线上 HTML 和资源的实际字节。

《Photograph》的 4:16 裁切、歌词和时间轴是该曲的内容设定；新歌曲分别使用自己的时长和素材。修改公共布局或播放器后，应同时回归参考页和至少一首不同名称、时长的生成页。
