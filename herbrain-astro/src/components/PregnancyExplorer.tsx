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
  const [week, setWeek] = useState(15);
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
    <div className="space-y-4">
      {/* Compact Header with Overview */}
      <div className="bg-white rounded-xl border border-gray-200 p-4">
        <div className="flex items-center gap-4 mb-3">
          <img
            src="/assets/pregnancy_logo.png"
            alt="Pregnancy"
            className="w-12 h-12"
          />
          <h1 className="text-xl font-semibold text-herbrain-dark">
            Digital Twin of the Pregnant Brain
          </h1>
        </div>
        <p className="text-sm text-herbrain-muted leading-relaxed">
          Explore AI-predicted shape changes of subcortical brain structures during pregnancy.{' '}
          <span className="text-red-500 font-medium">Red</span> = growing,{' '}
          <span className="text-blue-500 font-medium">blue</span> = shrinking,{' '}
          <span style={{ color: '#B5A08A' }} className="font-medium">beige</span> = pre-pregnancy baseline.
        </p>
      </div>

      {/* Week Slider - Inline */}
      <div className="bg-white rounded-xl border border-gray-200 px-6 py-3">
        <div className="max-w-md mx-auto">
          <WeekSlider
            value={week}
            onChange={setWeek}
            label="Gestational Week"
          />
        </div>
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Animation Explorer - Left Column */}
        <div className="lg:col-span-3">
          <AnimationExplorer week={week} />
        </div>

        {/* MRI Viewer - Middle Column */}
        <div className="lg:col-span-4">
          <MriViewer week={week} />
        </div>

        {/* Mesh Explorer - Right Column */}
        <div className="lg:col-span-5">
          <MeshExplorerSimple week={week} containerRef={meshContainerRef} />
        </div>
      </div>

      {/* GPT Chat */}
      <GptChat week={week} getMeshScreenshot={getMeshScreenshot} />
    </div>
  );
}

/**
 * Simplified MeshExplorer that only displays, without its own slider
 * (week is controlled by parent)
 */
function MeshExplorerSimple({ week, containerRef }: { week: number; containerRef?: React.RefObject<HTMLDivElement | null> }) {
  const [meshData, setMeshData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Dynamic import to avoid SSR issues with Plotly
  const [Plot, setPlot] = useState<any>(null);

  useEffect(() => {
    // Dynamically import react-plotly.js
    import('react-plotly.js').then((mod) => {
      setPlot(() => mod.default);
    });
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
    return figure?.data || [];
  };

  const getLayout = () => ({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    width: 480,
    height: 380,
    scene: {
      aspectmode: 'data',
      xaxis: { visible: false, showgrid: false },
      yaxis: { visible: false, showgrid: false },
      zaxis: { visible: false, showgrid: false },
      bgcolor: 'rgba(250, 251, 252, 0)',
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    uirevision: 'constant',
  });

  if (loading || !Plot) {
    return (
      <div className="flex flex-col items-center justify-center h-80 bg-white rounded-xl border border-gray-200 p-6">
        <div className="loading-spinner mb-3"></div>
        <p className="text-sm text-herbrain-muted">Loading brain visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-80 bg-white rounded-xl border border-gray-200 p-6">
        <p className="text-red-500 text-sm mb-3">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-3 py-1.5 text-sm bg-herbrain-green text-white rounded-lg hover:bg-herbrain-green/90"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex flex-col items-center bg-white rounded-xl border border-gray-200 p-4">
      <div className="mb-2">
        <Plot
          data={getPlotData()}
          layout={getLayout()}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            displaylogo: false,
            responsive: true,
          }}
        />
      </div>

      {/* Color Legend - Compact */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-red-500"></div>
          <span className="text-herbrain-muted">Growing</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded bg-blue-500"></div>
          <span className="text-herbrain-muted">Shrinking</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded" style={{ backgroundColor: '#E5D4C0' }}></div>
          <span className="text-herbrain-muted">Baseline</span>
        </div>
      </div>
    </div>
  );
}

