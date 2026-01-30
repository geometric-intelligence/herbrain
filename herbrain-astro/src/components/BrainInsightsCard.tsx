import { useState, useMemo } from 'react';

interface BrainInsightsCardProps {
  week: number;
}

// Research-backed content for each trimester/period
const INSIGHTS_CONTENT = {
  firstTrimester: {
    title: "Your Brain is Beginning to Transform",
    content: [
      "Your body is now producing up to **1,000 times more hormones** than usual — including estrogen and progesterone.",
      "These powerful hormones are already triggering changes in your brain, preparing it for the incredible journey of motherhood.",
      "Think of it as your brain's way of getting ready to bond with and care for your baby.",
    ],
    keyFact: "Brain plasticity — your brain's ability to change and adapt — is now in high gear.",
  },
  secondTrimester: {
    title: "Your Brain is Actively Remodeling",
    content: [
      "By now, your brain's gray matter has changed by about **2.7%**. This is completely normal and healthy.",
      "Meanwhile, the **white matter** connections between brain regions are getting stronger, improving communication pathways.",
      "These changes are most active in areas linked to **understanding others**, **social bonding**, and **emotional processing** — all skills you'll need as a new mom.",
    ],
    keyFact: "Your brain is fine-tuning itself for your new role, especially in regions for empathy and social cognition.",
  },
  thirdTrimester: {
    title: "Your Brain is Preparing for Motherhood",
    content: [
      "Your brain has now changed by about **5%** — the most dramatic brain remodeling since adolescence.",
      "Key areas like the **hippocampus** (memory and learning) and **hypothalamus** (bonding hormones) are adapting for your new role.",
      "The regions changing most are in the **Default Mode Network** and **Frontoparietal Network** — involved in thinking about yourself, others, and planning for the future.",
    ],
    keyFact: "These changes help your brain become more attuned to your baby's needs after birth.",
  },
  postpartum: {
    title: "Your Brain is Recovering and Adapting",
    content: [
      "After birth, your brain begins a gradual **recovery process** — gray matter volume starts to increase again (about 3.4% in the first 6 months).",
      "However, not all changes reverse. Some adaptations persist for **years** — perhaps permanently — shaping you as a mother.",
      "Research shows that mothers with greater brain recovery report **stronger attachment** to their babies.",
    ],
    keyFact: "Your brain continues to adapt postpartum, and recovery is linked to maternal well-being and bonding.",
  },
};

// Papers to cite
const RESEARCH_PAPERS = [
  {
    authors: "Pritschet et al.",
    year: "2024",
    journal: "Nature Neuroscience",
    title: "Neuroanatomical changes observed over the course of a human pregnancy",
    url: "https://doi.org/10.1038/s41593-024-01741-0",
  },
  {
    authors: "Servin-Barthet et al.",
    year: "2025",
    journal: "Nature Communications",
    title: "Pregnancy entails a U-shaped trajectory in human brain structure",
    url: "https://doi.org/10.1038/s41467-025-55830-0",
  },
];

function formatContent(text: string) {
  // Convert **text** to bold spans
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} className="text-herbrain-dark font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return part;
  });
}

export default function BrainInsightsCard({ week }: BrainInsightsCardProps) {
  const [showSources, setShowSources] = useState(false);

  // Determine which content to show based on week
  const insights = useMemo(() => {
    if (week <= 12) return INSIGHTS_CONTENT.firstTrimester;
    if (week <= 27) return INSIGHTS_CONTENT.secondTrimester;
    if (week <= 40) return INSIGHTS_CONTENT.thirdTrimester;
    return INSIGHTS_CONTENT.postpartum;
  }, [week]);

  return (
    <div className="premium-card-static flex flex-col h-full min-h-0">
      {/* Header with dynamic title */}
      <div className="px-4 sm:px-5 lg:px-6 pt-3 sm:pt-4 pb-2 sm:pb-3 border-b border-herbrain-border/40 flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-lg bg-herbrain-green/10 flex items-center justify-center flex-shrink-0">
            <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-herbrain-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h2 className="text-xs sm:text-sm lg:text-base font-semibold text-herbrain-dark leading-tight">
            {insights.title}
          </h2>
        </div>
      </div>

      {/* Content - Scrollable with responsive font sizing */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-4 sm:px-5 lg:px-6 py-2 sm:py-3 lg:py-4">
        {/* Main content paragraphs - smaller min font on constrained spaces */}
        <div className="space-y-2 lg:space-y-3">
          {insights.content.map((paragraph, idx) => (
            <p key={idx} className="text-xs sm:text-sm lg:text-base text-herbrain-muted leading-relaxed">
              {formatContent(paragraph)}
            </p>
          ))}
        </div>

        {/* Key Fact Highlight - scales proportionally */}
        <div className="mt-3 lg:mt-4 p-2.5 lg:p-3 rounded-xl bg-herbrain-green/5 border border-herbrain-green/10">
          <div className="flex items-start gap-2">
            <svg className="w-3.5 h-3.5 lg:w-4 lg:h-4 text-herbrain-green flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-[11px] sm:text-xs lg:text-sm text-herbrain-dark leading-relaxed">
              {formatContent(insights.keyFact)}
            </p>
          </div>
        </div>
      </div>

      {/* Footer - Science Sources */}
      <div className="px-4 sm:px-6 py-3 border-t border-herbrain-border/40 flex-shrink-0">
        <button
          onClick={() => setShowSources(!showSources)}
          className="flex items-center gap-1.5 text-xs text-herbrain-muted hover:text-herbrain-green transition-colors"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          <span>Where does the science come from?</span>
          <svg 
            className={`w-3 h-3 transition-transform ${showSources ? 'rotate-180' : ''}`} 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Expandable sources list */}
        {showSources && (
          <div className="mt-3 space-y-2 animate-fade-in">
            {RESEARCH_PAPERS.map((paper, idx) => (
              <a
                key={idx}
                href={paper.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-2.5 rounded-lg bg-herbrain-surface/50 hover:bg-herbrain-surface transition-colors group"
              >
                <p className="text-xs font-medium text-herbrain-dark group-hover:text-herbrain-green transition-colors">
                  {paper.authors} ({paper.year})
                </p>
                <p className="text-[10px] text-herbrain-muted mt-0.5 line-clamp-1">
                  {paper.title}
                </p>
                <p className="text-[10px] text-herbrain-muted/70 italic">
                  {paper.journal}
                </p>
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
