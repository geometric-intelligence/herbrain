# HerBrain Astro

A static Astro website for exploring brain changes during pregnancy, converted from the original Dash application.

## Features

- **3D Mesh Visualization**: Interactive Plotly.js visualization of subcortical brain structures
- **MRI Viewer**: Lazy-loaded NIfTI files from Cloudflare R2 storage
- **Pregnancy Animation**: Video player with frame-based seeking
- **AI Chat**: GPT-4o powered neurobot for answering questions about brain changes
- **Responsive Design**: Mobile-friendly layout with collapsible sidebar and adaptive visualizations

## Quick Start

```bash
# Install dependencies
make install

# Start development server
make dev

# Build for production
make build

# Preview production build
make preview
```

## Project Structure

```
herbrain-astro/
├── src/
│   ├── components/       # React components
│   │   ├── MeshExplorer.tsx
│   │   ├── MriViewer.tsx
│   │   ├── AnimationExplorer.tsx
│   │   ├── GptChat.tsx
│   │   └── ...
│   ├── layouts/          # Astro layouts
│   ├── lib/              # Utility functions
│   ├── pages/            # Astro pages
│   └── styles/           # Global CSS
├── public/
│   ├── assets/           # Static assets (logos, video)
│   └── data/             # Pre-computed mesh data
├── scripts/              # Build scripts
└── Makefile
```

## Pre-computing Mesh Data

The mesh predictions are pre-computed from the Python models:

```bash
cd scripts
python precompute_meshes.py --data-dir /path/to/data --output ../public/data/prerendered_meshes.json
```

## MRI Storage (Cloudflare R2)

MRI NIfTI files are lazy-loaded from Cloudflare R2:

1. Create R2 bucket:
   ```bash
   npx wrangler r2 bucket create herbrain-mri
   ```

2. Upload NIfTI files:
   ```bash
   ./scripts/upload_mri.sh /path/to/mri/data
   ```

3. Enable public access:
   ```bash
   npx wrangler r2 bucket public-access set herbrain-mri --enable
   ```

4. Update `public/data/metadata.json` with your R2 bucket URL.

## Deployment

Deploy to Cloudflare Pages:

```bash
make deploy
```

Or manually:

```bash
npm run build
npx wrangler pages deploy dist --project-name herbrain
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PUBLIC_R2_BUCKET_URL` | Base URL for R2 bucket containing MRI files |

## GPT Chat

The AI chat feature uses client-side OpenAI API calls. Users must provide their own API key, which is stored locally in the browser's localStorage.

## Development

- **Astro**: Static site generator
- **React**: UI components (via @astrojs/react)
- **Tailwind CSS**: Styling
- **Plotly.js**: 3D mesh visualization
- **nifti-reader-js**: NIfTI file parsing

## License

See [LICENSE.md](../LICENSE.md) in the parent directory.
