import { useState, useEffect, useRef, useCallback } from 'react';
import AnimationExplorer from './AnimationExplorer';
import MriViewer from './MriViewer';
import WeekSlider from './WeekSlider';
import BrainInsightsCard from './BrainInsightsCard';

/**
 * Main pregnancy explorer component that combines all visualization components
 * with shared gestational week state.
 */
export default function PregnancyExplorer() {
  const [week, setWeek] = useState(20);
  const meshContainerRef = useRef<HTMLDivElement>(null);
  
  // Mobile carousel state
  const [activeCard, setActiveCard] = useState(0);
  const cardLabels = ['3D Brain', 'Journey', 'MRI Scan'];
  
  // Lifted state for brain overlay toggle (shared between mobile and desktop)
  const [showBrainOverlay, setShowBrainOverlay] = useState(true);

  return (
    <div className="flex flex-col h-[calc(100vh-2rem)] md:h-[calc(100vh-3rem)] animate-fade-in overflow-hidden">
      {/* Hero Header - Simplified for mobile */}
      <header className="premium-card-static p-3 md:p-5 flex-shrink-0">
        <div className="flex items-center gap-3 md:gap-5">
          <div className="relative flex-shrink-0">
            <div className="absolute inset-0 bg-herbrain-green/10 rounded-2xl blur-xl"></div>
            <img
              src="/assets/pregnancy_logo.png"
              alt="Pregnancy"
              className="relative w-8 h-8 md:w-12 md:h-12 drop-shadow-sm"
            />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-base md:text-xl font-semibold text-herbrain-dark tracking-tight leading-tight">
              Your Brain's Digital Twin During Pregnancy
            </h1>
            <p className="hidden md:block text-sm text-herbrain-muted mt-1 leading-relaxed">
              Watch how your brain transforms week by week. Move the timeline below to see AI-predicted changes in real time.
            </p>
          </div>
        </div>
      </header>

      {/* Mobile Layout - Only visible on small screens */}
      <div className="md:hidden flex flex-col flex-1 min-h-0 gap-2 mt-2">
        {/* Mobile Carousel - Responsive height based on screen */}
        <div className="relative h-[45vh] min-h-[320px] max-h-[420px] flex-shrink-0">
          {/* 3D Brain Model - Card 0 */}
          <div className={`absolute inset-0 bg-white rounded-2xl overflow-hidden transition-all duration-300 ${activeCard === 0 ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <MeshExplorerSimple 
              week={week} 
              containerRef={meshContainerRef} 
              isMobile={true}
              showBrainOverlay={showBrainOverlay}
              onShowBrainOverlayChange={setShowBrainOverlay}
            />
          </div>
          
          {/* Journey - Card 1 */}
          <div className={`absolute inset-0 bg-white rounded-2xl overflow-hidden transition-all duration-300 ${activeCard === 1 ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <AnimationExplorer week={week} />
          </div>
          
          {/* MRI Scan - Card 2 */}
          <div className={`absolute inset-0 bg-white rounded-2xl overflow-hidden transition-all duration-300 ${activeCard === 2 ? 'opacity-100 z-10' : 'opacity-0 z-0 pointer-events-none'}`}>
            <MriViewer week={week} />
          </div>
        </div>

        {/* Mobile Navigation Arrows */}
        <div className="flex items-center justify-center gap-6 flex-shrink-0">
          <button
            onClick={() => setActiveCard((prev) => (prev - 1 + 3) % 3)}
            className="p-2 rounded-full bg-white border border-herbrain-border shadow-sm hover:shadow-md transition-all active:scale-95"
            aria-label="Previous card"
          >
            <svg className="w-4 h-4 text-herbrain-dark" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          
          {/* Dots Indicator with Label */}
          <div className="flex flex-col items-center gap-1">
            <div className="flex items-center gap-2">
              {cardLabels.map((label, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveCard(idx)}
                  className={`transition-all duration-200 ${
                    activeCard === idx 
                      ? 'w-6 h-1.5 rounded-full bg-herbrain-green' 
                      : 'w-1.5 h-1.5 rounded-full bg-herbrain-border hover:bg-herbrain-muted/50'
                  }`}
                  aria-label={`Go to ${label}`}
                />
              ))}
            </div>
            <span className="text-[10px] text-herbrain-muted">{cardLabels[activeCard]}</span>
          </div>
          
          <button
            onClick={() => setActiveCard((prev) => (prev + 1) % 3)}
            className="p-2 rounded-full bg-white border border-herbrain-border shadow-sm hover:shadow-md transition-all active:scale-95"
            aria-label="Next card"
          >
            <svg className="w-4 h-4 text-herbrain-dark" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Mobile Brain Controls - Only visible when 3D Brain card is active */}
        {activeCard === 0 && (
          <div className="flex flex-col items-center gap-2 px-4 py-2 bg-white rounded-xl border border-herbrain-border/60 mx-4 flex-shrink-0">
            {/* Brain Overlay Toggle */}
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showBrainOverlay}
                onChange={(e) => setShowBrainOverlay(e.target.checked)}
                className="w-4 h-4 rounded border-herbrain-border text-herbrain-green focus:ring-herbrain-green/30 cursor-pointer"
              />
              <span className="text-xs text-herbrain-muted">Show Full Brain</span>
            </label>

            {/* Compact Legend */}
            <div className="flex items-center gap-3 flex-wrap justify-center">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-gradient-to-br from-red-400 to-red-500"></span>
                <span className="text-[10px] text-herbrain-muted">Growing</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-gradient-to-br from-blue-400 to-blue-500"></span>
                <span className="text-[10px] text-herbrain-muted">Shrinking</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: 'linear-gradient(135deg, #E5D4C0 0%, #D4C4B0 100%)' }}></span>
                <span className="text-[10px] text-herbrain-muted">Baseline</span>
              </div>
            </div>
          </div>
        )}

        {/* Brain Insights Card - Fills remaining space (only this scrolls on mobile) */}
        <div className="flex-1 min-h-0">
          <BrainInsightsCard week={week} />
        </div>

        {/* Week Slider - Fixed at bottom, full width */}
        <div className="premium-card-static px-4 py-2 flex-shrink-0">
          <WeekSlider
            value={week}
            onChange={setWeek}
            label="Gestational Week"
            hideWeek40={true}
            compact={true}
          />
        </div>
      </div>

      {/* Desktop Layout - Hidden on mobile */}
      <div className="hidden md:flex md:flex-col flex-1 min-h-0 gap-4 mt-4">
        {/* Desktop Grid - Responsive height visualization row */}
        <div className="grid md:grid-cols-2 lg:grid-cols-12 gap-4 flex-shrink-0 h-[42vh] min-h-[340px] max-h-[500px]">
          {/* Animation Explorer - Narrower Left Column */}
          <div className="md:col-span-1 lg:col-span-2 h-full overflow-hidden">
            <AnimationExplorer week={week} />
          </div>

          {/* MRI Viewer - Middle Column */}
          <div className="md:col-span-1 lg:col-span-4 h-full overflow-hidden">
            <MriViewer week={week} />
          </div>

          {/* Mesh Explorer - Right Column - Prominent */}
          <div className="md:col-span-2 lg:col-span-6 h-full">
            <MeshExplorerSimple 
              week={week} 
              containerRef={meshContainerRef}
              showBrainOverlay={showBrainOverlay}
              onShowBrainOverlayChange={setShowBrainOverlay}
            />
          </div>
        </div>

        {/* Brain Insights Card - Dynamic height based on available space */}
        <div className="flex-1 min-h-[120px]">
          <BrainInsightsCard week={week} />
        </div>

        {/* Week Slider - Full width, compact height */}
        <div className="premium-card-static px-4 sm:px-6 py-2 flex-shrink-0">
          <WeekSlider
            value={week}
            onChange={setWeek}
            label="Gestational Week"
            hideWeek40={true}
          />
        </div>
      </div>
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
function MeshExplorerSimple({ 
  week, 
  containerRef, 
  isMobile = false,
  showBrainOverlay,
  onShowBrainOverlayChange,
}: { 
  week: number; 
  containerRef?: React.RefObject<HTMLDivElement | null>; 
  isMobile?: boolean;
  showBrainOverlay: boolean;
  onShowBrainOverlayChange: (value: boolean) => void;
}) {
  const [meshData, setMeshData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Hover tooltip state
  const [hoveredStructure, setHoveredStructure] = useState<string | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const plotContainerRef = useRef<HTMLDivElement>(null);
  
  // Responsive sizing
  const [chartDimensions, setChartDimensions] = useState({ width: 480, height: 340 });

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

  // Responsive chart sizing
  useEffect(() => {
    const container = plotContainerRef.current;
    if (!container) return;

    const updateDimensions = () => {
      const rect = container.getBoundingClientRect();
      const width = Math.max(280, Math.min(rect.width - 20, 600));
      // Calculate height based on width ratio but cap it to ensure controls are visible
      const idealHeight = Math.round(width * 0.65); // Reduced from 0.7 to leave more room
      // Cap height to ensure there's always space for controls (max 280px for smaller screens)
      const maxChartHeight = isMobile ? 300 : 280;
      const height = Math.min(idealHeight, maxChartHeight);
      setChartDimensions({ width, height });
    };

    updateDimensions();
    
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(container);
    
    return () => resizeObserver.disconnect();
  }, [isMobile]);

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
    width: chartDimensions.width,
    height: chartDimensions.height,
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
    <div ref={containerRef} className="viz-card flex flex-col h-full p-4 sm:p-5 overflow-hidden">
      <h2 className="text-xs sm:text-sm font-semibold text-herbrain-dark uppercase tracking-wide mb-3 flex-shrink-0">
        3D Brain Model
      </h2>
      
      <div ref={plotContainerRef} className="flex-1 flex items-center justify-center plotly-container relative overflow-hidden min-h-0">
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

      {/* Brain Overlay Toggle + Legend - Hidden on mobile carousel */}
      {!isMobile && (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-0 pt-3 border-t border-herbrain-border/40 flex-shrink-0">
          {/* Brain Overlay Toggle */}
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showBrainOverlay}
              onChange={(e) => onShowBrainOverlayChange(e.target.checked)}
              className="w-4 h-4 rounded border-herbrain-border text-herbrain-green focus:ring-herbrain-green/30 cursor-pointer"
            />
            <span className="text-xs sm:text-sm text-herbrain-muted">Show Full Brain</span>
          </label>

          {/* Compact Legend */}
          <div className="flex items-center gap-3 sm:gap-4 flex-wrap">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-gradient-to-br from-red-400 to-red-500"></span>
              <span className="text-[10px] sm:text-xs text-herbrain-muted">Growing</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-gradient-to-br from-blue-400 to-blue-500"></span>
              <span className="text-[10px] sm:text-xs text-herbrain-muted">Shrinking</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full" style={{ background: 'linear-gradient(135deg, #E5D4C0 0%, #D4C4B0 100%)' }}></span>
              <span className="text-[10px] sm:text-xs text-herbrain-muted">Baseline</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
