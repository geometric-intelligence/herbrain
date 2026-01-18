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
    <div className="space-y-8">
      {/* Banner */}
      <div className="flex items-center gap-4">
        <img
          src="/assets/pregnancy_logo.png"
          alt="Pregnancy"
          className="w-16 h-16"
        />
        <h1 className="text-2xl font-semibold text-herbrain-dark">
          Digital Twin of the Pregnant Brain
        </h1>
      </div>

      {/* Overview */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-herbrain-dark mb-3">Overview</h2>
        <p className="text-herbrain-muted leading-relaxed">
          The subcortical structures of the brain are sensitive to sex hormone changes.
          In pregnancy, hormones experience extreme fluctuations, and subcortical
          structure volumes are known to decrease. However, we find that the shape of
          these structures change as well. We have trained an AI to predict shape
          changes of the subcortical structures based on gestational week.{' '}
          <span className="text-blue-600">Blue</span> areas indicate growth and{' '}
          <span className="text-red-600">red</span> areas indicate shrinkage compared
          to pre-pregnancy shape.{' '}
          <span style={{ color: '#B5A08A' }}>Beige</span> color indicates
          pre-pregnancy shape.
        </p>
      </div>

      {/* Instructions */}
      <div className="bg-herbrain-green/5 border border-herbrain-green/20 rounded-xl p-4">
        <p className="text-herbrain-dark">
          <strong>Instructions:</strong> Change the gestational week slider below, and the
          AI model will predict subcortical structure shape changes for that week.
          The MRI view and animation will update to show the corresponding data.
        </p>
      </div>

      {/* Shared Week Slider */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="max-w-lg mx-auto">
          <WeekSlider
            value={week}
            onChange={setWeek}
            label="Gestational Week"
          />
        </div>
      </div>

      {/* Main Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Animation Explorer - Left Column */}
        <div className="lg:col-span-3">
          <AnimationExplorer week={week} />
        </div>

        {/* MRI Viewer - Middle Column */}
        <div className="lg:col-span-4">
          <MriViewer week={week} />
        </div>

        {/* Mesh Explorer - Right Column (no internal slider, uses shared week) */}
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
  const [showBrain, setShowBrain] = useState(true);

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
    width: 550,
    height: 450,
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
      <div className="flex flex-col items-center justify-center h-96 bg-white rounded-xl border border-gray-200 p-8">
        <div className="loading-spinner mb-4"></div>
        <p className="text-herbrain-muted">Loading brain visualization...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 bg-white rounded-xl border border-gray-200 p-8">
        <p className="text-red-500 mb-4">{error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-herbrain-green text-white rounded-lg hover:bg-herbrain-green/90"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex flex-col items-center bg-white rounded-xl border border-gray-200 p-6">
      <div className="mb-4">
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

      {/* Toggle for whole brain view */}
      <div className="flex items-center gap-3 mb-4">
        <input
          type="checkbox"
          id="show-brain-simple"
          checked={showBrain}
          onChange={(e) => setShowBrain(e.target.checked)}
          className="w-4 h-4 text-herbrain-green rounded focus:ring-herbrain-green"
        />
        <label htmlFor="show-brain-simple" className="text-sm text-herbrain-dark">
          Show Full Brain Overlay
        </label>
      </div>

      {/* Color Legend */}
      <div className="flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-red-500"></div>
          <span className="text-herbrain-muted">Shrinking</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-blue-500"></div>
          <span className="text-herbrain-muted">Growing</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E5D4C0' }}></div>
          <span className="text-herbrain-muted">Unchanged</span>
        </div>
      </div>
    </div>
  );
}

