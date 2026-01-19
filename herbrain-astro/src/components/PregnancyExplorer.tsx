import { useState, useEffect, useRef, useCallback } from 'react';
import AnimationExplorer from './AnimationExplorer';
import MriViewer from './MriViewer';
import GptChat from './GptChat';
import WeekSlider from './WeekSlider';

/**
 * Main pregnancy explorer component that combines all visualization components
 * with shared gestational week state.
 */
export default function PregnancyExplorer() {
  const [week, setWeek] = useState(20);
  const meshContainerRef = useRef<HTMLDivElement>(null);

  // Screenshot capture function for GPT chat
  const getMeshScreenshot = useCallback(async (): Promise<string | null> => {
    try {
      const plotDiv = meshContainerRef.current?.querySelector('.js-plotly-plot') as HTMLElement | null;
      if (!plotDiv) {
        console.warn('Could not find Plotly chart element');
        return null;
      }
      
      // Dynamically import Plotly for toImage
      const Plotly = await import('plotly.js-dist-min');
      const dataUrl = await Plotly.default.toImage(plotDiv, {
        format: 'png',
        width: 800,
        height: 600,
      });
      return dataUrl;
    } catch (err) {
      console.warn('Failed to capture mesh screenshot:', err);
      return null;
    }
  }, []);

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Hero Header */}
      <header className="premium-card-static p-6">
        <div className="flex items-center gap-5">
          <div className="relative flex-shrink-0">
            <div className="absolute inset-0 bg-herbrain-green/10 rounded-2xl blur-xl"></div>
            <img
              src="/assets/pregnancy_logo.png"
              alt="Pregnancy"
              className="relative w-14 h-14 drop-shadow-sm"
            />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-semibold text-herbrain-dark tracking-tight">
              Your Brain's Digital Twin During Pregnancy
            </h1>
            <p className="text-base text-herbrain-muted mt-1.5 leading-relaxed">
              Watch how your brain transforms week by week. Move the timeline below to see AI-predicted changes in real time.
            </p>
          </div>
        </div>
      </header>

      {/* Main Visualization Grid - Fixed height cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4" style={{ minHeight: '420px' }}>
        {/* Animation Explorer - Narrower Left Column */}
        <div className="lg:col-span-2">
          <AnimationExplorer week={week} />
        </div>

        {/* MRI Viewer - Middle Column */}
        <div className="lg:col-span-4">
          <MriViewer week={week} />
        </div>

        {/* Mesh Explorer - Right Column - Prominent */}
        <div className="lg:col-span-6">
          <MeshExplorerSimple week={week} containerRef={meshContainerRef} />
        </div>
      </div>

      {/* Week Slider - YouTube style, below cards */}
      <div className="premium-card-static px-6 py-5">
        <WeekSlider
          value={week}
          onChange={setWeek}
          label="Gestational Week"
        />
      </div>

      {/* GPT Chat */}
      <GptChat week={week} getMeshScreenshot={getMeshScreenshot} />
    </div>
  );
}

// Subcortical structure info: names and brief descriptions
const STRUCTURE_INFO: Record<string, { name: string; description: string }> = {
'L_Thal': { name: 'Left Thalamus', description: 'Sensory and motor relay, consciousness, and sleep regulation' },
  'R_Thal': { name: 'Right Thalamus', description: 'Sensory and motor relay, consciousness, and sleep regulation' },
  'L_Caud': { name: 'Left Caudate', description: 'Motor planning, goal-directed behavior, and learning' },
  'R_Caud': { name: 'Right Caudate', description: 'Motor planning, goal-directed behavior, and learning' },
  'L_Puta': { name: 'Left Putamen', description: 'Regulation of movement and procedural learning' },
  'R_Puta': { name: 'Right Putamen', description: 'Regulation of movement and procedural learning' },
  'L_Pall': { name: 'Left Pallidum', description: 'Regulation of voluntary movement and inhibitory control' },
  'R_Pall': { name: 'Right Pallidum', description: 'Regulation of voluntary movement and inhibitory control' },
  'L_Hipp': { name: 'Left Hippocampus', description: 'Memory consolidation, spatial navigation, and learning' },
  'R_Hipp': { name: 'Right Hippocampus', description: 'Memory consolidation, spatial navigation, and learning' },
  'L_Amyg': { name: 'Left Amygdala', description: 'Emotional processing, fear conditioning, and threat detection' },
  'R_Amyg': { name: 'Right Amygdala', description: 'Emotional processing, fear conditioning, and threat detection' },
  'L_Accu': { name: 'Left Accumbens', description: 'Reward processing, pleasure, and motivation' },
  'R_Accu': { name: 'Right Accumbens', description: 'Reward processing, pleasure, and motivation' },
};

/**
 * Simplified MeshExplorer that only displays, without its own slider
 * (week is controlled by parent)
 */
function MeshExplorerSimple({ week, containerRef }: { week: number; containerRef?: React.RefObject<HTMLDivElement | null> }) {
  const [meshData, setMeshData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showBrainOverlay, setShowBrainOverlay] = useState(true);
  
  // Hover tooltip state
  const [hoveredStructure, setHoveredStructure] = useState<string | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const plotContainerRef = useRef<HTMLDivElement>(null);

  // Dynamic import to avoid SSR issues with Plotly
  const [Plot, setPlot] = useState<any>(null);

  useEffect(() => {
    // Dynamically import react-plotly.js
    import('react-plotly.js').then((mod) => {
      setPlot(() => mod.default);
    });
  }, []);

  // Track mouse position within the plot container
  useEffect(() => {
    const container = plotContainerRef.current;
    if (!container) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    };

    container.addEventListener('mousemove', handleMouseMove);
    return () => container.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const { loadMeshData } = await import('../lib/meshData');
        const data = await loadMeshData();
        setMeshData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load mesh data:', err);
        setError('Failed to load brain visualization data');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const getPlotData = () => {
    if (!meshData) return [];
    
    // Find nearest available week
    const availableWeeks = Object.keys(meshData).map(Number).sort((a, b) => a - b);
    let nearestWeek = availableWeeks[0];
    let minDiff = Math.abs(week - nearestWeek);
    for (const w of availableWeeks) {
      const diff = Math.abs(week - w);
      if (diff < minDiff) {
        minDiff = diff;
        nearestWeek = w;
      }
    }
    
    const figure = meshData[String(nearestWeek)];
    let data = figure?.data || [];
    
    // Filter out brain_overlay trace if toggle is off
    if (!showBrainOverlay) {
      data = data.filter((trace: any) => trace.name !== 'brain_overlay');
    }
    
    return data;
  };

  const getLayout = () => ({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    width: 480,
    height: 340,
    scene: {
      aspectmode: 'data',
      xaxis: { visible: false, showgrid: false },
      yaxis: { visible: false, showgrid: false },
      zaxis: { visible: false, showgrid: false },
      bgcolor: 'rgba(248, 250, 251, 0)',
      camera: {
        eye: { x: 0.8, y: 0.8, z: 0.6 },  // More zoomed in
        center: { x: 0, y: 0, z: 0 },
      },
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    uirevision: 'constant',
  });

  // Handle hover on mesh structures
  const handleHover = useCallback((event: any) => {
    if (event.points && event.points.length > 0) {
      const point = event.points[0];
      const traceName = point.data?.name;
      
      // Only show tooltip for subcortical structures (not brain_overlay)
      if (traceName && traceName !== 'brain_overlay' && STRUCTURE_INFO[traceName]) {
        setHoveredStructure(traceName);
      }
    }
  }, []);

  const handleUnhover = useCallback(() => {
    setHoveredStructure(null);
  }, []);

  if (loading || !Plot) {
    return (
      <div className="viz-card flex flex-col h-full items-center justify-center p-6">
        <div className="loading-spinner mb-3"></div>
        <p className="text-base text-herbrain-muted">Loading 3D visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="viz-card flex flex-col h-full items-center justify-center p-6">
        <p className="text-red-500 text-base mb-3">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="premium-btn-secondary text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="viz-card flex flex-col h-full p-5">
      <h2 className="text-sm font-semibold text-herbrain-dark uppercase tracking-wide mb-3">
        3D Brain Model
      </h2>
      
      <div ref={plotContainerRef} className="flex-1 flex items-center justify-center plotly-container relative">
        <Plot
          data={getPlotData()}
          layout={getLayout()}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'select2d', 'lasso2d'],
            displaylogo: false,
            responsive: true,
          }}
          onHover={handleHover}
          onUnhover={handleUnhover}
        />
        
        {/* Floating Tooltip - follows cursor within plot area */}
        {hoveredStructure && STRUCTURE_INFO[hoveredStructure] && (
          <div
            className="absolute z-50 pointer-events-none"
            style={{
              left: mousePosition.x + 16,
              top: mousePosition.y + 12,
            }}
          >
            <div
              className="px-3.5 py-2.5 rounded-lg"
              style={{
                background: 'rgba(15, 23, 42, 0.95)',  // herbrain-dark (#0F172A)
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                boxShadow: '0 4px 16px rgba(0, 0, 0, 0.25)',
                animation: 'fadeIn 0.1s ease-out',
                maxWidth: '260px',
              }}
            >
              <div className="text-[13px] font-semibold text-white leading-tight">
                {STRUCTURE_INFO[hoveredStructure].name}
              </div>
              <div className="text-[11px] text-white/70 mt-1 leading-relaxed">
                {STRUCTURE_INFO[hoveredStructure].description}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Brain Overlay Toggle + Legend */}
      <div className="flex items-center justify-between pt-3 border-t border-herbrain-border/40">
        {/* Brain Overlay Toggle */}
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showBrainOverlay}
            onChange={(e) => setShowBrainOverlay(e.target.checked)}
            className="w-4 h-4 rounded border-herbrain-border text-herbrain-green focus:ring-herbrain-green/30 cursor-pointer"
          />
          <span className="text-sm text-herbrain-muted">Show Full Brain</span>
        </label>

        {/* Compact Legend */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-br from-red-400 to-red-500"></span>
            <span className="text-xs text-herbrain-muted">Growing</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-gradient-to-br from-blue-400 to-blue-500"></span>
            <span className="text-xs text-herbrain-muted">Shrinking</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'linear-gradient(135deg, #E5D4C0 0%, #D4C4B0 100%)' }}></span>
            <span className="text-xs text-herbrain-muted">Baseline</span>
          </div>
        </div>
      </div>
    </div>
  );
}
