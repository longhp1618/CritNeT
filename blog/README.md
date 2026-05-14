# CritNeT — standalone blog page

A self-contained, single-page write-up of the CritNeT toolkit, designed to be deployed independently of any other site.

```
CritNeT_blog/
├── index.html          # the page
├── style.css           # all styling (no build step required)
├── figures/            # PNGs (see below)
└── README.md           # this file
```

## What's inside

- **Vanilla HTML + CSS** — no framework, no build step, no JavaScript framework. Just open `index.html` in a browser.
- **KaTeX** for equations (CDN-loaded, supports `$..$`, `$$..$$`, `\(..\)`, `\[..\]`).
- **Prism.js** for Python syntax highlighting (CDN-loaded with autoloader).
- **System / Google fonts**: Source Serif 4 (body), Inter (UI), JetBrains Mono (code).
- **One accent colour** (`--color-accent: #1d3a8a`) — change it in `style.css` to retheme.
- **Sticky TOC** on screens ≥ 1180 px wide; collapses gracefully on mobile.

## Figures

The HTML references three figures under `figures/`:

| Source PDF (in repo)                                              | Target PNG                                          |
|-------------------------------------------------------------------|-----------------------------------------------------|
| `CritNeT_Latex/figures/utility_ratios.pdf`                        | `figures/utility_ratios.png`                        |
| `CritNeT_Latex/figures/grid_search_Llama-3.1-8B-Instruct.pdf`     | `figures/grid_search_Llama-3.1-8B-Instruct.png`     |
| `CritNeT_Latex/figures/grid_search_Qwen3-4B-Instruct.pdf`         | `figures/grid_search_Qwen3-4B-Instruct.png`         |

Convert them with whichever tool you have around:

```bash
# poppler-utils
pdftoppm -png -r 220 CritNeT_Latex/figures/utility_ratios.pdf \
                     CritNeT_blog/figures/utility_ratios
# → produces CritNeT_blog/figures/utility_ratios-1.png; rename without the -1.

# ImageMagick
magick -density 220 CritNeT_Latex/figures/utility_ratios.pdf \
       CritNeT_blog/figures/utility_ratios.png

# Or simply embed the PDFs by changing the <img src="..."> values in index.html.
```

220 DPI is plenty for retina screens; bump higher if you want absolute sharpness.

## Local preview

Any static server works:

```bash
cd CritNeT_blog
python3 -m http.server 8000          # then open http://localhost:8000
# or
npx serve .
```

(Opening `index.html` directly with `file://` works too, but some browsers
block fonts / CDN resources on `file://` URLs.)

## Deployment

Drop the folder onto any static host:

- **GitHub Pages.** Push `CritNeT_blog/` to a repo, enable Pages on the branch, point at `/CritNeT_blog`.
- **Netlify / Vercel / Cloudflare Pages.** Drag-and-drop the folder, or point at the repo.
- **Custom domain.** All assets are relative paths, so the folder root just needs to be served.

If you want the page at `your-domain.com/critnet/`, move the folder to that path on your server — no rewrites needed.

## Customisation hot-spots

- **Accent colour.** `style.css`, top of file (`--color-accent`). Try `#7c2d12` for burgundy, `#0f766e` for teal, `#7e22ce` for purple.
- **Fonts.** Change the `@import`-equivalent `<link>` URL near the top of `index.html`, then the `--font-*` variables in `style.css`.
- **Brand link.** The "CritNeT" brand mark in the site header points to `#top`. Repoint it to `https://longhp1618.github.io` or wherever you want.
- **Citation.** Update the BibTeX block at the bottom of `index.html` once the paper has an arXiv id.
- **Author info / affiliation.** In the `.article-byline` block near the top of `index.html`.

## Print-friendliness

The default styles render reasonably with the browser print dialog. If you need a polished PDF export, consider adding `@media print` overrides.

## License

Same as the parent repository.
