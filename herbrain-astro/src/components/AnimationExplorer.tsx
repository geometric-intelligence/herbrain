import { useRef, useEffect } from 'react';

interface AnimationExplorerProps {
  week: number;
}

// Fetus size comparisons by gestational week
const FETUS_SIZE_BY_WEEK: Record<number, string> = {
  0: 'a poppy seed',
  1: 'a poppy seed',
  2: 'a poppy seed',
  3: 'a poppy seed',
  4: 'a poppy seed',
  5: 'a poppy seed',
  6: 'a poppy seed',
  7: 'a blueberry',
  8: 'a blueberry',
  9: 'a blueberry',
  10: 'a kumquat',
  11: 'a kumquat',
  12: 'a lime',
  13: 'a lime',
  14: 'a lime',
  15: 'an apple',
  16: 'an apple',
  17: 'an apple',
  18: 'an apple',
  19: 'an heirloom tomato',
  20: 'an heirloom tomato',
  21: 'an heirloom tomato',
  22: 'an heirloom tomato',
  23: 'a large mango',
  24: 'a large mango',
  25: 'a rutabaga',
  26: 'a rutabaga',
  27: 'a rutabaga',
  28: 'a large eggplant',
  29: 'a large eggplant',
  30: 'a large eggplant',
  31: 'a coconut',
  32: 'a coconut',
  33: 'a coconut',
  34: 'a coconut',
  35: 'a honeydew melon',
  36: 'a honeydew melon',
  37: 'a honeydew melon',
  38: 'a honeydew melon',
  39: 'a honeydew melon',
  40: 'a small pumpkin',
};

function getFetusSize(week: number): string {
  const clampedWeek = Math.max(0, Math.min(40, Math.round(week)));
  return FETUS_SIZE_BY_WEEK[clampedWeek] || 'a small pumpkin';
}

export default function AnimationExplorer({ week }: AnimationExplorerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  const getFrameTime = (gestWeek: number): number => {
    if (gestWeek >= 41) return 9;
    if (gestWeek >= 40) return 8;
    return Math.floor(gestWeek / 5);
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = getFrameTime(week) + 0.001;
  }, [week]);

  const handleLoadedData = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = getFrameTime(week) + 0.001;
  };

  return (
    <div className="viz-card flex flex-col h-full overflow-hidden p-5">
      <h2 className="text-sm font-semibold text-herbrain-dark uppercase tracking-wide mb-3">
        Journey
      </h2>
      
      {/* Video Container */}
      <div className="flex-1 flex items-center justify-center">
        <video
          ref={videoRef}
          src="/assets/pregnancy_animation.mp4"
          preload="auto"
          muted
          playsInline
          onLoadedData={handleLoadedData}
          className="max-h-48 w-auto object-contain"
        />
      </div>
      
      {/* Fetus size - Highlighted */}
      <div className="pt-3 border-t border-herbrain-border/30 mt-auto">
        <p className="text-xs text-center text-herbrain-muted/80">
          Baby is the size of
        </p>
        <div className="flex justify-center mt-1.5">
          <span 
            className="inline-block px-3 py-1.5 text-herbrain-green text-base font-medium rounded-full text-center"
            style={{ backgroundColor: 'rgb(61 122 107 / 0.08)' }}
          >
            {getFetusSize(week)}
          </span>
        </div>
      </div>
    </div>
  );
}
