# UT3C Heritage

A collaborative platform for collecting, sharing, and curating 3D captures of heritage places and spatial designs. Built with Next.js, React, and Three.js.

## Features

- **3D PLY Viewer**: Interactive 3D visualization tool powered by Three.js
  - Advanced camera controls (WASD + Space/Shift for movement)
  - Double-click to set pivot points
  - ArcballControls for smooth rotation
  - Support for both point cloud and mesh rendering

- **Project Gallery**: Showcase heritage preservation projects with detailed information
- **Responsive Design**: Mobile-friendly interface with modern UI
- **Navigation**: Intuitive navigation between pages

## Color Scheme

- Background: White (#ffffff)
- Accent: Light Blue (#60a5fa)
- Text: Black (#000000)

## Project Structure

```
ut3c-heritage/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Landing page
│   │   ├── projects/         # Projects gallery
│   │   ├── about/            # About page
│   │   ├── viewer/           # 3D PLY viewer page
│   │   ├── layout.tsx        # Root layout
│   │   └── globals.css       # Global styles
│   └── components/
│       ├── Navigation.tsx    # Navigation component
│       ├── ProjectCard.tsx   # Project card component
│       └── PLYViewer.tsx     # 3D PLY viewer component
├── public/
│   └── auditorio.ply        # Sample PLY file
└── package.json
```

## Getting Started

### Prerequisites

- Node.js 18+ 
- pnpm (recommended) or npm

### Installation

1. Install dependencies:
```bash
pnpm install
```

2. Place your PLY files in the `public/` directory

3. Run the development server:
```bash
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

### Adding New Projects

Edit `src/app/page.tsx` or `src/app/projects/page.tsx` to add new project cards:

```tsx
<ProjectCard 
  title="Your Project Name"
  location="Location"
  description="Project description"
  viewerUrl="/viewer"
/>
```

### Adding PLY Files

1. Place your `.ply` file in the `public/` directory
2. Update the PLY loader path in `src/components/PLYViewer.tsx`:

```tsx
loader.load('/your-file.ply', ...)
```

## 3D Viewer Controls

- **Mouse**: Click and drag to rotate
- **W/A/S/D**: Move forward/left/backward/right
- **Space**: Move up
- **Shift**: Move down
- **Double-click**: Set new pivot point on surface

## Technologies

- **Next.js 15**: React framework for production
- **React 19**: UI library
- **Three.js**: 3D graphics library
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework

## Inspired By

This platform is inspired by [MIT Design Heritage](https://designheritage.mit.edu/), a collaborative platform for heritage preservation and education.

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
