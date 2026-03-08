// Types
export interface Paper {
  id: string;
  title: string;
  description: string;
  source: string;
  url: string;
  published_at: string;
  fetched_at: string;
  authors: string;
  topics: string[];
  ranking_score: number;
  status: 'unread' | 'read' | 'starred';
}

export interface Digest {
  date: string;
  top_paper_id: string;
  papers_fetched: number;
  pdf_path: string;
  whatsapp_sent: boolean;
  workflow_run_id: string;
}

export interface Config {
  keywords: string[];
  sources: {
    arxiv: boolean;
    huggingface: boolean;
    pwc: boolean;
    blogs: boolean;
  };
  schedule: string;
  whatsapp_enabled: boolean;
}

// GitHub raw content URL for fetching data
const GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/vickey-kapoor/ai-research-whatsapp-digest/master/data';

async function fetchJsonFile(filename: string): Promise<unknown | null> {
  try {
    const response = await fetch(`${GITHUB_RAW_BASE}/${filename}`, {
      cache: 'no-store' // Always fetch fresh data
    });
    if (response.ok) {
      return await response.json();
    }
  } catch (error) {
    console.error(`Failed to fetch ${filename}:`, error);
  }
  return null;
}

export async function getPapers(): Promise<Paper[]> {
  const data = await fetchJsonFile('papers.json') as { papers?: Paper[] } | null;
  return data?.papers || [];
}

export async function getDigests(): Promise<Digest[]> {
  const data = await fetchJsonFile('digests.json') as { digests?: Digest[] } | null;
  return data?.digests || [];
}

export async function getConfig(): Promise<Config> {
  const data = await fetchJsonFile('config.json') as Config | null;

  return data || {
    keywords: [
      'AI agent', 'autonomous agent', 'reasoning', 'chain of thought',
      'CoT', 'ReAct', 'tool use', 'planning', 'multi-agent', 'agentic'
    ],
    sources: {
      arxiv: true,
      huggingface: true,
      pwc: true,
      blogs: true
    },
    schedule: '0 16 * * *',
    whatsapp_enabled: true
  };
}

export async function getReportDates(): Promise<string[]> {
  // Derive report dates from digests
  const digests = await getDigests();
  const dates = digests
    .filter(d => d.pdf_path)
    .map(d => {
      const match = d.pdf_path.match(/reports\/([^/]+)\//);
      return match ? match[1] : null;
    })
    .filter((date): date is string => date !== null);

  return [...new Set(dates)].sort().reverse();
}

// Stats helpers
export async function getStats() {
  const papers = await getPapers();
  const digests = await getDigests();

  const today = new Date().toISOString().split('T')[0];
  const todaysPapers = papers.filter(p => p.fetched_at?.startsWith(today));
  const todaysDigest = digests.find(d => d.date === today);

  const sourceCounts = papers.reduce((acc, p) => {
    acc[p.source] = (acc[p.source] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return {
    totalPapers: papers.length,
    todaysPapers: todaysPapers.length,
    totalDigests: digests.length,
    todaysDigest,
    sourceCounts,
    unreadCount: papers.filter(p => p.status === 'unread').length,
    starredCount: papers.filter(p => p.status === 'starred').length,
  };
}
// Trigger redeploy Sat, Mar  7, 2026  6:24:18 PM
