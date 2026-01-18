import { useRef, useEffect } from 'react';

interface AnimationExplorerProps {
  week: number;
}

/**
 * AnimationExplorer component that displays a pregnancy animation video
 * with frame seeking based on gestational week.
 * 
 * The video has 10 frames (weeks 00, 05, 10, 15, 20, 25, 30, 35, 40, 41) at 1fps.
 * Frame seeking happens entirely in the browser - no network requests.
 */
export default function AnimationExplorer({ week }: AnimationExplorerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Map gestational week to video time (frame number at 1fps)
  const getFrameTime = (gestWeek: number): number => {
    if (gestWeek >= 41) {
      return 9;
    } else if (gestWeek >= 40) {
      return 8;
    } else {
      return Math.floor(gestWeek / 5);
    }
  };

  // Seek video to correct frame when week changes
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const frameTime = getFrameTime(week);
    // Add small offset to ensure we're in the frame
    video.currentTime = frameTime + 0.001;
  }, [week]);

  // Initial load handler
  const handleLoadedData = () => {
    const video = videoRef.current;
    if (!video) return;

    // Seek to initial frame
    const frameTime = getFrameTime(week);
    video.currentTime = frameTime + 0.001;
  };

  return (
    <div className="flex flex-col items-center justify-center h-full">
      <div className="relative bg-white rounded-xl border border-gray-200 p-4 overflow-hidden">
        <video
          ref={videoRef}
          src="/assets/pregnancy_animation.mp4"
          preload="auto"
          muted
          playsInline
          onLoadedData={handleLoadedData}
          className="max-h-96 w-auto object-contain"
          style={{ display: 'block', margin: '0 auto' }}
        />
        
        {/* Week indicator */}
        <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-lg shadow-sm">
          <span className="text-sm font-medium text-herbrain-dark">
            Week {week}
          </span>
        </div>
      </div>
      
      {/* Caption */}
      <p className="text-sm text-herbrain-muted mt-3 text-center max-w-xs">
        Pregnancy progression visualization showing body and brain changes
      </p>
    </div>
  );
}
