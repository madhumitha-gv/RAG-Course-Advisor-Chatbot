export type Source = { sourceNum: number; program: string; heading: string; file: string }
export type AdvisorMessage = { role: 'user' | 'assistant'; content: string; sources?: Source[] }
export type QueryResult = { answer: string; sources: Source[] }

type ApiSource = {
  source_num: number
  program: string
  heading: string
  file: string
}

type ApiQueryResult = { answer: string; sources: ApiSource[] }

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function queryAdvisor(question: string): Promise<QueryResult> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
  }

  const result = (await response.json()) as ApiQueryResult
  return {
    answer: result.answer,
    sources: result.sources.map((source) => ({
      sourceNum: source.source_num,
      program: source.program,
      heading: source.heading,
      file: source.file,
    })),
  }
}