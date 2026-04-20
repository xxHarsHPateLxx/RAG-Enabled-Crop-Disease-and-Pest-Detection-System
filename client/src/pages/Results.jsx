import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertTriangle, Leaf, CheckCircle, MessageCircle, Send } from 'lucide-react';

const SESSION_STORAGE_KEY = 'agrisentry_session_id';
const RESULT_STORAGE_KEY = 'agrisentry_last_result';

const readStoredResult = () => {
  try {
    const rawResult = window.localStorage.getItem(RESULT_STORAGE_KEY);
    return rawResult ? JSON.parse(rawResult) : null;
  } catch {
    return null;
  }
};

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const [image, setImage] = useState(location.state?.image ?? null);
  const [result, setResult] = useState(location.state?.result ?? readStoredResult());
  const [followUp, setFollowUp] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState('');

  useEffect(() => {
    if (location.state?.result) {
      setResult(location.state.result);
      window.localStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(location.state.result));
    }

    if (location.state?.image) {
      setImage(location.state.image);
    }
  }, [location.state]);

  useEffect(() => {
    if (!result) {
      navigate('/');
    }
  }, [result, navigate]);

  if (!result) {
    return null;
  }

  const { crop, disease, confidence, advice, reasoning, follow_up_questions: followUpQuestions = [], analysis_source: analysisSource, prediction_mode: predictionMode, session_id: sessionId } = result;

  const modeLabel = predictionMode === 'multimodal_llm' ? 'Multimodal LLM' : 'CNN + LLM';

  const hasReasoningContent = Boolean(
    reasoning?.summary
      || reasoning?.key_factors?.length
      || reasoning?.evidence_points?.length
      || reasoning?.safety_notes?.length
  );

  const renderInlineMarkdown = (text, keyPrefix) => {
    if (!text) {
      return null;
    }

    const tokens = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g).filter(Boolean);
    return tokens.map((token, index) => {
      const key = `${keyPrefix}-${index}`;

      if (token.startsWith('**') && token.endsWith('**')) {
        return <strong key={key} className="font-semibold text-gray-900">{token.slice(2, -2)}</strong>;
      }

      if (token.startsWith('*') && token.endsWith('*')) {
        return <em key={key} className="italic text-gray-800">{token.slice(1, -1)}</em>;
      }

      if (token.startsWith('`') && token.endsWith('`')) {
        return <code key={key} className="rounded bg-gray-100 px-1.5 py-0.5 text-[0.9em] text-gray-800">{token.slice(1, -1)}</code>;
      }

      return <React.Fragment key={key}>{token}</React.Fragment>;
    });
  };

  const renderMarkdownText = (text) => {
    if (!text) return null;

    const elements = [];
    let currentIndex = 0;
    const lines = text.split('\n');
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) {
        i += 1;
        continue;
      }

      const h1Match = line.match(/^#\s+\*\*(.+?)\*\*\s*$/);
      if (h1Match) {
        elements.push(
          <h2 key={`h1-${currentIndex++}`} className="text-2xl font-bold text-emerald-900 mt-7 mb-3 first:mt-0 border-b border-emerald-200 pb-2">
            {renderInlineMarkdown(h1Match[1], `h1-${currentIndex}`)}
          </h2>
        );
        i += 1;
        continue;
      }

      const h2Match = line.match(/^##\s+\*\*(.+?)\*\*\s*$/);
      if (h2Match) {
        elements.push(
          <h3 key={`h2-${currentIndex++}`} className="text-xl font-bold text-emerald-800 mt-6 mb-2 first:mt-0">
            {renderInlineMarkdown(h2Match[1], `h2-${currentIndex}`)}
          </h3>
        );
        i += 1;
        continue;
      }

      const headingMatch = line.match(/^\*\*(.+?)\*\*:?[\s]*$/);
      if (headingMatch) {
        elements.push(
          <h4 key={`heading-${currentIndex++}`} className="text-lg font-semibold text-emerald-800 mt-5 mb-2 first:mt-0">
            {renderInlineMarkdown(headingMatch[1], `heading-${currentIndex}`)}
          </h4>
        );
        i += 1;
        continue;
      }

      if (line.trim().startsWith('-') || line.trim().startsWith('•')) {
        const bulletItems = [];
        while (i < lines.length && (lines[i].trim().startsWith('-') || lines[i].trim().startsWith('•'))) {
          let bulletText = lines[i].trim().substring(1).trim();
          bulletText = bulletText.replace(/\*\*(.+?)\*\*/g, '$1');

          if (bulletText.endsWith(':')) {
            if (bulletItems.length > 0) {
              elements.push(
                <ul key={`list-${currentIndex++}`} className="list-disc list-inside ml-4 mb-4 space-y-2">
                  {bulletItems}
                </ul>
              );
              bulletItems.length = 0;
            }

            elements.push(
              <h5 key={`subheading-${currentIndex++}`} className="text-base font-semibold text-emerald-700 mt-3 mb-1">
                {renderInlineMarkdown(bulletText, `sub-${currentIndex}`)}
              </h5>
            );
            i += 1;
          } else {
            bulletItems.push(
              <li key={`bullet-${currentIndex++}`} className="mb-1.5 text-gray-700 marker:text-emerald-500">
                {renderInlineMarkdown(bulletText, `bullet-${currentIndex}`)}
              </li>
            );
            i += 1;
          }
        }

        if (bulletItems.length > 0) {
          elements.push(
            <ul key={`list-${currentIndex++}`} className="list-disc list-inside ml-4 mb-4 space-y-1.5">
              {bulletItems}
            </ul>
          );
        }
        continue;
      }

      if (line.trim().match(/^\d+\./)) {
        const numberedItems = [];
        while (i < lines.length && lines[i].trim().match(/^\d+\./)) {
          let numberText = lines[i].trim().replace(/^\d+\./, '').trim();
          numberText = numberText.replace(/\*\*(.+?)\*\*/g, '$1');
          numberedItems.push(
            <li key={`number-${currentIndex++}`} className="mb-1.5 text-gray-700 marker:font-semibold marker:text-emerald-600">
              {renderInlineMarkdown(numberText, `num-${currentIndex}`)}
            </li>
          );
          i += 1;
        }

        elements.push(
          <ol key={`numlist-${currentIndex++}`} className="list-decimal list-inside ml-4 mb-4 space-y-1.5">
            {numberedItems}
          </ol>
        );
        continue;
      }

      let cleanText = line.trim();
      cleanText = cleanText.replace(/^#+\s+/g, '');
      cleanText = cleanText.replace(/\*\*(.+?)\*\*/g, '$1');

      if (cleanText) {
        elements.push(
          <p key={`para-${currentIndex++}`} className="text-gray-700 leading-relaxed mb-2.5">
            {renderInlineMarkdown(cleanText, `para-${currentIndex}`)}
          </p>
        );
      }

      i += 1;
    }

    return elements;
  };

  const handleTryAnother = () => {
    navigate('/');
  };

  const handleBack = () => {
    navigate(-1);
  };

  const handleFollowUp = async (nextMessage = followUp) => {
    if (!nextMessage.trim()) {
      return;
    }

    const activeSessionId = sessionId || window.localStorage.getItem(SESSION_STORAGE_KEY);
    if (!activeSessionId) {
      setChatError('No session is available for follow-up. Run a diagnosis first.');
      return;
    }

    setChatLoading(true);
    setChatError('');
    setChatHistory((currentHistory) => [...currentHistory, { role: 'user', content: nextMessage }]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSessionId, message: nextMessage }),
      });

      if (!response.ok) {
        throw new Error('Follow-up request failed');
      }

      const data = await response.json();
      setChatHistory((currentHistory) => [...currentHistory, { role: 'assistant', content: data.answer }]);
      setFollowUp('');
    } catch {
      setChatError('Unable to get a follow-up answer right now.');
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-6 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-emerald-50 via-white to-amber-50">
      <div className="max-w-7xl mx-auto">
        <button
          onClick={handleBack}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 mb-6 font-medium"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back</span>
        </button>

        <div className="bg-white/90 rounded-2xl shadow-xl overflow-hidden border border-green-100 backdrop-blur">
          <div className="bg-gradient-to-r from-green-700 via-emerald-700 to-green-600 px-6 py-5">
            <h1 className="text-3xl font-bold text-white text-center">Analysis Results</h1>
            <p className="text-green-100 text-center mt-2">Multimodal diagnosis, retrieval, and follow-up memory</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
            <div className="space-y-3 lg:col-span-5">
              <h2 className="text-xl font-semibold text-green-800">Analyzed Image</h2>
              {image ? (
                <img
                  src={image}
                  alt="Analyzed crop"
                  className="w-full h-80 rounded-xl shadow-lg bg-white border border-green-100 object-contain"
                />
              ) : (
                <div className="w-full h-80 rounded-xl border border-dashed border-green-200 bg-green-50 flex items-center justify-center text-green-700">
                  Image preview unavailable after reload.
                </div>
              )}
            </div>

            <div className="space-y-4 lg:col-span-7">
              <div className="bg-green-50 p-6 rounded-xl">
                <div className="flex items-center space-x-3 mb-4">
                  <Leaf className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-green-800">Crop</h3>
                </div>
                <p className="text-2xl font-bold text-green-700">{crop}</p>
                <p className="text-green-600 mt-1">Pipeline: {modeLabel}</p>
                <p className="text-green-600 mt-1">Analysis source: {analysisSource}</p>
              </div>

              <div className="bg-red-50 p-6 rounded-xl border border-red-200">
                <div className="flex items-center space-x-3 mb-4">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
                  <h3 className="text-lg font-semibold text-red-800">Detected Issue</h3>
                </div>
                <p className="text-2xl font-bold text-red-700">{disease}</p>
                <p className="text-red-600 mt-1">Confidence: {(confidence * 100).toFixed(1)}%</p>
              </div>

              {reasoning && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                    <p className="text-xs uppercase tracking-wider text-emerald-700">Severity</p>
                    <p className="mt-2 text-lg font-semibold text-emerald-900">{reasoning.severity_assessment}</p>
                  </div>
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
                    <p className="text-xs uppercase tracking-wider text-amber-700">Confidence band</p>
                    <p className="mt-2 text-lg font-semibold text-amber-900">{reasoning.confidence_band}</p>
                  </div>
                  <div className="rounded-xl border border-sky-200 bg-sky-50 p-4">
                    <p className="text-xs uppercase tracking-wider text-sky-700">Intervention mode</p>
                    <p className="mt-2 text-lg font-semibold text-sky-900">{reasoning.intervention_mode}</p>
                  </div>
                  <div className="rounded-xl border border-lime-200 bg-lime-50 p-4">
                    <p className="text-xs uppercase tracking-wider text-lime-700">Priority</p>
                    <p className="mt-2 text-lg font-semibold text-lime-900">{reasoning.intervention_priority}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 px-6 pb-6">
            {hasReasoningContent && (
              <div className="bg-white rounded-xl border border-green-100 shadow-sm p-5 space-y-4 lg:col-span-5">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                  <h2 className="text-xl font-bold text-gray-800">Internal Reasoning Summary</h2>
                </div>

                {reasoning?.summary && <p className="text-gray-700">{reasoning.summary}</p>}

                {reasoning?.key_factors?.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-2">Key factors</h3>
                    <ul className="list-disc list-inside space-y-1.5 text-gray-700 marker:text-green-500">
                      {reasoning.key_factors.map((factor) => (
                        <li key={factor}>{factor}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {reasoning?.evidence_points?.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-2">Evidence points</h3>
                    <ul className="list-disc list-inside space-y-1.5 text-gray-700 marker:text-green-500">
                      {reasoning.evidence_points.map((point) => (
                        <li key={point}>{point}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {reasoning?.safety_notes?.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-2">Safety notes</h3>
                    <ul className="list-disc list-inside space-y-1.5 text-gray-700 marker:text-green-500">
                      {reasoning.safety_notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <div className={`bg-gradient-to-br from-blue-50 to-green-50 p-5 rounded-xl border border-green-200 shadow-inner space-y-4 ${hasReasoningContent ? 'lg:col-span-7' : 'lg:col-span-12'}`}>
              <div className="flex items-center gap-3 mb-2">
                <CheckCircle className="h-7 w-7 text-green-600" />
                <h2 className="text-2xl font-bold text-gray-800">AI-Powered Analysis & Recommendations</h2>
              </div>
              <div className="max-w-none rounded-lg bg-white/60 border border-emerald-100 p-4">
                {renderMarkdownText(advice)}
              </div>
            </div>
          </div>

          <div className="px-6 pb-6 space-y-5">
            {followUpQuestions.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <MessageCircle className="h-6 w-6 text-amber-700" />
                  <h2 className="text-xl font-semibold text-amber-900">Suggested follow-up questions</h2>
                </div>
                <div className="flex flex-wrap gap-3">
                  {followUpQuestions.map((question) => (
                    <button
                      key={question}
                      onClick={() => handleFollowUp(question)}
                      className="rounded-full border border-amber-300 bg-white px-4 py-2 text-sm font-medium text-amber-900 hover:bg-amber-100"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl border border-green-100 p-5 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <MessageCircle className="h-6 w-6 text-green-600" />
                <h2 className="text-xl font-bold text-gray-800">Ask a follow-up</h2>
              </div>

              <div className="space-y-3">
                <textarea
                  value={followUp}
                  onChange={(event) => setFollowUp(event.target.value)}
                  rows={3}
                  placeholder="Ask about urgency, substitutes, dosage timing, or what to do next..."
                  className="w-full rounded-xl border border-gray-300 px-4 py-3 text-gray-700 focus:border-green-500 focus:outline-none"
                />
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleFollowUp()}
                    disabled={chatLoading}
                    className="inline-flex items-center gap-2 rounded-xl bg-green-600 px-5 py-3 font-semibold text-white hover:bg-green-700 disabled:bg-gray-400"
                  >
                    <Send className="h-4 w-4" />
                    {chatLoading ? 'Sending...' : 'Send follow-up'}
                  </button>
                  {chatError && <p className="text-sm text-red-600">{chatError}</p>}
                </div>
              </div>

              {chatHistory.length > 0 && (
                <div className="mt-4 space-y-3">
                  {chatHistory.map((message, index) => (
                    <div
                      key={`${message.role}-${index}`}
                      className={`rounded-xl p-4 ${message.role === 'user' ? 'bg-green-50' : 'bg-gray-50'}`}
                    >
                      <p className="text-xs uppercase tracking-wider text-gray-500 mb-2">{message.role}</p>
                      <div className="prose prose-sm max-w-none text-gray-700">{renderMarkdownText(message.content)}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="bg-gray-50 px-6 py-5 text-center">
            <button
              onClick={handleTryAnother}
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 shadow-lg"
            >
              Try Another Image
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
