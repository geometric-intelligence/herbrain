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
  5: 'a sesame seed',
  6: 'a lentil',
  7: 'a blueberry',
  8: 'a raspberry',
  9: 'a grape',
  10: 'a kumquat',
  11: 'a fig',
  12: 'a lime',
  13: 'a lemon',
  14: 'a peach',
  15: 'an apple',
  16: 'an avocado',
  17: 'a pear',
  18: 'a bell pepper',
  19: 'a mango',
  20: 'a banana',
  21: 'a carrot',
  22: 'a papaya',
  23: 'a grapefruit',
  24: 'an ear of corn',
  25: 'a cauliflower',
  26: 'a lettuce head',
  27: 'a cabbage',
  28: 'an eggplant',
  29: 'a butternut squash',
  30: 'a coconut',
  31: 'a pineapple',
  32: 'a squash',
  33: 'a durian',
  34: 'a cantaloupe',
  35: 'a honeydew melon',
  36: 'a romaine lettuce',
  37: 'a winter melon',
  38: 'a leek',
  39: 'a mini watermelon',
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
    <div className="viz-card flex flex-col h-full overflow-hidden">
      <h2 className="text-sm font-semibold text-herbrain-dark uppercase tracking-wide px-4 pt-4 pb-2">
        Journey
      </h2>
      
      {/* Video Container */}
      <div className="flex-1 flex items-center justify-center px-2 py-2">
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
      <div className="px-3 pb-4 pt-3 border-t border-herbrain-border/30 mt-auto">
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
