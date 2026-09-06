import { useState } from 'react'
import type { FormEvent } from 'react'
import { queryAdvisor, type AdvisorMessage, type Source } from './api'
import './App.css'

const programs = [
  { name: 'Computer Science', detail: 'MS / PhD', count: 3 },
  { name: 'Data Science', detail: 'Residential MS', count: 2 },
  { name: 'Informatics', detail: 'MS / PhD', count: 2 },
  { name: 'Intelligent Systems', detail: 'MS / PhD', count: 1 },
  { name: 'Secure Computing', detail: 'MS', count: 1 },
]

const suggestedQuestions = [
  'What are the core courses for the Data Science MS?',
  'Compare the thesis and non-thesis paths for Informatics.',
  'Which machine learning courses are available as electives?',
]

function App() {
  const [messages, setMessages] = useState<AdvisorMessage[]>([])
  const [question, setQuestion] = useState('')
  const [isThinking, setIsThinking] = useState(false)

  async function submitQuestion(event?: FormEvent) {
    event?.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || isThinking) return

    setMessages((current) => [...current, { role: 'user', content: trimmedQuestion }])
    setQuestion('')
    setIsThinking(true)
    try {
      const result = await queryAdvisor(trimmedQuestion)
      setMessages((current) => [...current, { role: 'assistant', content: result.answer, sources: result.sources }])
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The advisor service is unavailable.'
      setMessages((current) => [...current, { role: 'assistant', content: `I could not reach the handbook service. ${message}` }])
    } finally {
      setIsThinking(false)
    }
  }

  function selectSuggestion(suggestion: string) {
    setQuestion(suggestion)
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">IU</div><div><strong>Luddy</strong><span>Course Advisor</span></div></div>
        <div className="sidebar-rule" />
        <div className="sidebar-label">Programs</div>
        <nav className="program-list" aria-label="Programs">
          {programs.map((program, index) => <button className={`program ${index === 1 ? 'active' : ''}`} key={program.name} type="button"><span className="program-dot" /><span className="program-copy"><strong>{program.name}</strong><small>{program.detail}</small></span><span className="program-count">{program.count}</span></button>)}
        </nav>
        <div className="sidebar-footer"><div className="status-dot" /><div><strong>Handbook library</strong></div></div>
      </aside>

      <section className="workspace">
        <div className="conversation" aria-live="polite">
          {messages.length === 0 ? <section className="empty-state"><div className="orbit" aria-hidden="true"><span>?</span></div><p className="section-kicker">Official program handbooks</p><h2>Clear answers for<br /><em>your next move.</em></h2><p className="empty-copy">Search requirements, compare programs, and find your path through Luddy graduate study.</p><div className="suggestions">{suggestedQuestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => selectSuggestion(suggestion)}><span>{suggestion}</span><b>&#8599;</b></button>)}</div></section> : <div className="message-list">{messages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}`}><div className="message-avatar">{message.role === 'user' ? 'You' : 'LA'}</div><div className="message-body"><div className="message-meta">{message.role === 'user' ? 'You' : 'Luddy Advisor'}</div><p>{message.content}</p>{message.sources?.length ? <SourceList sources={message.sources} /> : null}</div></article>)}{isThinking && <div className="thinking"><span /><span /><span /> Searching the handbook</div>}</div>}
        </div>
        <form className="composer-wrap" onSubmit={submitQuestion}><div className="composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submitQuestion() } }} placeholder="Ask about courses, requirements, or programs..." rows={1} aria-label="Your question" /><button className="send-button" type="submit" disabled={!question.trim() || isThinking} aria-label="Send question" title="Send question">&#8593;</button></div><span className="composer-note">Answers are grounded in official Luddy handbooks. Verify decisions with your academic advisor.</span></form>
      </section>
    </main>
  )
}

function SourceList({ sources }: { sources: Source[] }) {
  return <div className="sources"><div className="sources-heading"><span>Sources</span><span>{sources.length} references</span></div>{sources.map((source) => <div className="source" key={source.sourceNum}><span className="source-number">0{source.sourceNum}</span><div><strong>{source.program}</strong><span>{source.heading} / {source.file}</span></div><span className="source-arrow">&#8599;</span></div>)}</div>
}

export default App
