
import React, { useState, useEffect, useRef } from 'react';
import { Upload, Sun, Moon, Send, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import EquiCADIcon from './assets/bita.png';
interface ChatMessage {
  id: string;
  type: 'bot' | 'user';
  text: string;
  options?: Array<{ id: string; text: string }>;
  preview?: string;
  result?: any;
  timestamp: Date;
}

interface AnalysisResult {
  chunk_number: number;
  content_preview: string;
  result: string;
  error: boolean;
}

const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [textInputEnabled, setTextInputEnabled] = useState(false);
  const [fileUploadEnabled, setFileUploadEnabled] = useState(false);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [allResults, setAllResults] = useState<AnalysisResult[]>([]);
  const conversationStartedRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [textError, setTextError] = useState<string | null>(null);
  const MAX_WORDS = 200;

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  useEffect(() => {
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode) {
      setDarkMode(savedMode === 'true');
    }

    if (!conversationStartedRef.current) {
      conversationStartedRef.current = true;
      startConversation();
    }
  }, []);


  useEffect(() => {
    localStorage.setItem('darkMode', darkMode.toString());
  }, [darkMode]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };



  const startConversation = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/chat/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      
      if (data.success) {
        setSessionId(data.session_id);
        addBotMessage(data.message, data.options);
      } else {
        addBotMessage('Failed to start conversation. Please refresh the page.');
      }
    } catch (error) {
      console.error('Failed to start conversation:', error);
      addBotMessage('Unable to connect to the server. Please ensure the backend is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const addBotMessage = (text: string, options?: any[], preview?: string, result?: any) => {
    const message: ChatMessage = {
      id: `bot-${Date.now()}-${Math.random()}`,
      type: 'bot',
      text,
      options,
      preview,
      result,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, message]);
  };

  const addUserMessage = (text: string) => {
    const message: ChatMessage = {
      id: `user-${Date.now()}-${Math.random()}`,
      type: 'user',
      text,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, message]);
  };

  const handleOptionClick = async (optionId: string, optionText: string, isTextEnabled: boolean = textInputEnabled) => {
    addUserMessage(optionText);
    setLoading(true);
    setTextInputEnabled(false);
    setFileUploadEnabled(false);

    try {
      if (optionId === 'upload_file') {
        if (!sessionId) {
          addBotMessage('Session not ready yet. Please wait a moment and try again.');
          setLoading(false);
          return;
        }

        setFileUploadEnabled(true);
        setLoading(false);
        addBotMessage('Please upload a PDF file:');
        return;
      }

      if (optionId === 'single_text') {
        if (!sessionId) {
          addBotMessage('Session is still initializing. Please wait a second.');
          setLoading(false);
          return;
        }

        setTextInputEnabled(true);
        setLoading(false);
        addBotMessage('Please enter the text you\'d like to analyze:');
        return;
      }


      // Handle content type selection=
      if (['all', 'text', 'table', 'figure'].includes(optionId)) {
        const response = await fetch(`${API_BASE_URL}/chat/select-content`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            content_type: optionId
          })
        });
        const data = await response.json();
        
        if (data.success) {
          addBotMessage(data.message, data.options);
        } else {
          addBotMessage(`Error: ${data.error || 'Something went wrong'}`);
        }
      }

      // Handle granularity selection
      else if (['sectional', 'paragraph', 'sentence'].includes(optionId)) {
        const response = await fetch(`${API_BASE_URL}/chat/select-granularity`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            granularity: optionId
          })
        });
        const data = await response.json();
        
        if (data.success) {
          addBotMessage(data.message, data.options);
        } else {
          addBotMessage(`Error: ${data.error || 'Something went wrong'}`);
        }
      }

      // Handle output format selection
      else if (['label', 'label_category'].includes(optionId)) {

      const endpoint = isTextEnabled ? '/chat/analyze-single' : '/chat/select-output-format';


      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          output_format: optionId
        })
      });

        const data = await response.json();
        
        if (data.success) {
          // Check if this is a single text  (has 'result' field)
          if (data.result) {
            // Single text analysis result
              addBotMessage(
              data.message, 
              data.options, 
              data.preview || '', // optional preview
              {
                type: data.result.includes('Label: Bias') ? 'bias' 
                      : data.result.includes('No label assigned') ? 'neutral'
                      : 'success',
                content: data.result
              }
            );
          } else if (data.ready_for_batch) {
            // File batch processing
            addBotMessage(data.message);
            setAllResults([]);
            setLoading(true);
            setFileUploadEnabled(false);
            setBatchProgress({ current: 0, total: data.total_chunks });
            setBatchProcessing(true);
            processBatch(0, data.total_chunks);
          } else {
            // Other response
            addBotMessage(data.message, data.options);
          }
        } else {
          addBotMessage(`Error: ${data.error || 'Something went wrong'}`);
          console.error('Output format error:', data);
        }
      }

      // Handle continuation choices
      else if (['yes', 'no', 'new_file', 'done'].includes(optionId)) {
        const response = await fetch(`${API_BASE_URL}/chat/continue`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            choice: optionId
          })
        });
        const data = await response.json();
        
        if (data.success) {
          addBotMessage(data.message, data.options);
          
          if (data.text_input_enabled) {
            setTextInputEnabled(true);
          }
          if (data.file_upload_enabled) {
            setFileUploadEnabled(true);
          }
          if (data.conversation_ended) { //restart conversation
            setTimeout(() => {
              setMessages([]);
              conversationStartedRef.current = false;
              startConversation();
            }, 2000);
          }
        } else {
          addBotMessage(`Error: ${data.error || 'Something went wrong'}`);
        }
      }

    } catch (error) {
      console.error('Error in handleOptionClick:', error);
      addBotMessage('Sorry, an error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const processBatch = async (chunkIndex: number, total: number) => {
  setBatchProcessing(true);

  try {
    const response = await fetch(`${API_BASE_URL}/chat/analyze-batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        chunk_index: chunkIndex
      })
    });

    const data = await response.json();

    if (data.success) {
      setAllResults(prev => [...prev, data.result]);

      // DISPLAY THE RESULT FOR THIS CHUNK
      addBotMessage(
        `Section ${data.result.chunk_number} Result:\n${data.result.result}`
      );

      setBatchProgress({
        current: chunkIndex + 1,
        total
      });

      if (!data.completed) {
        setTimeout(() => processBatch(chunkIndex + 1, total), 300);
      } else {
        setBatchProcessing(false);
        setLoading(false);

        addBotMessage(data.message, data.options);
      }
    } else {
      setBatchProcessing(false);
      addBotMessage(`Error: ${data.error || 'Batch processing failed'}`);
    }
  } catch (error) {
    setBatchProcessing(false);
    console.error(error);
  }
};



  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {

    if (loading || batchProcessing) {
      return;
    }

    if (!sessionId) {
      addBotMessage('Session not initialized. Please refresh the page.');
      return;
    }
    
    const file = e.target.files?.[0];
    if (!file) return;

      addBotMessage(`Uploading file: ${file.name}...`);



    if (!file.name.toLowerCase().endsWith('.pdf')) {
      addBotMessage('Please upload a PDF file only.');
      return;
    }

    addUserMessage(`Uploaded: ${file.name}`);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId || '');

      const response = await fetch(`${API_BASE_URL}/chat/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.success) {
        
        setFileUploadEnabled(false);
        
        addBotMessage(
          'PDF uploaded successfully. Preparing document for analysis…'
        );
        addBotMessage(data.message, data.options, data.preview);
      } else {
        addBotMessage(`Error: ${data.error}`);
      }
    } catch (error) {
        addBotMessage('Upload failed. Invalid session or server error. Please try again.');
        console.error('Upload error:', error);

    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleTextSubmit = async () => {
    if (!sessionId) {
      addBotMessage('Session not ready yet. Please wait a moment.');
      return;
    }

    const wordCount = textInput.trim().split(/\s+/).filter(Boolean).length;

    if (wordCount > MAX_WORDS) {
      setTextError(`Maximum ${MAX_WORDS} words allowed.`);
      return;
    }


    if (!textInput.trim()) return;

    const userText = textInput;
    addUserMessage(userText);
    setTextInput('');
    setLoading(true);
    setTextInputEnabled(false);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/single-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          text: userText
        })
      });
      const data = await response.json();

      if (data.success) {
        // Show output format options
        addBotMessage(data.message, data.options);
      } else {
        addBotMessage(`Error: ${data.error || 'Failed to process text'}`);
      }
    } catch (error) {
      addBotMessage('Error processing text. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleKeyPress = (e: React.KeyboardEvent, action: () => void) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      action();
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-900 via-slate-800 to-gray-900' 
        : 'bg-gradient-to-br from-gray-50 via-slate-100 to-gray-100'
    } flex items-center justify-center p-4`}>
      
      {/* Dark Mode Toggle */}
      <button type="button"
        onClick={toggleDarkMode}
        className={`fixed top-6 right-6 p-3 rounded-full shadow-lg transition-all duration-300 z-50 focus:outline-none focus:ring-4 ${
          darkMode
            ? 'bg-slate-700 hover:bg-slate-600 text-yellow-300 focus:ring-blue-500'
            : 'bg-white hover:bg-gray-100 text-slate-700 focus:ring-blue-400'
        }`}
        aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        tabIndex={0}
      >
        {darkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
      </button>

      {/* Main Chat Container */}
      <div className={`w-full max-w-4xl h-[90vh] rounded-2xl shadow-2xl overflow-hidden transition-colors duration-300 flex flex-col ${
        darkMode ? 'bg-slate-800' : 'bg-white'
      }`}>
        
        {/* Header */}
        <div className={`text-white p-6 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gradient-to-r from-slate-700 via-blue-900 to-slate-700' 
            : 'bg-gradient-to-r from-slate-700 via-blue-800 to-slate-700'
        }`}>
          <h1 className="text-2xl font-bold flex items-center">
            <img src={EquiCADIcon} alt="EquiCAD logo" className="w-7 h-7 mr-3" />
            EquiCAD: CAD Sex Bias Detection Assistant
          </h1>
          <p className={`text-sm mt-1 transition-colors duration-300 ${
            darkMode ? 'text-slate-300' : 'text-blue-100'
          }`}>
            Guided analysis of scientific text for sex bias in Coronary Artery Disease (CAD) research
          </p>
        </div>

        {/* Messages Area */}
        <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${
          darkMode ? 'bg-slate-900/50' : 'bg-gray-50'
        }`}>
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg p-4 transition-colors duration-300 ${
                msg.type === 'user'
                  ? darkMode
                    ? 'bg-blue-700 text-white'
                    : 'bg-blue-600 text-white'
                  : darkMode
                    ? 'bg-slate-700 text-slate-100'
                    : 'bg-white text-gray-900 shadow-md'
              }`}>
                <p className="whitespace-pre-wrap">{msg.text}</p>

                {msg.result && (
                  <div className={`mt-3 p-3 rounded border text-sm flex items-start ${
                    msg.result.type === 'bias'
                      ? darkMode
                        ? 'bg-yellow-900/20 border-yellow-600'
                        : 'bg-yellow-50 border-yellow-400'
                      : msg.result.type === 'neutral'
                        ? darkMode
                          ? 'bg-slate-800 border-slate-600'
                          : 'bg-gray-50 border-gray-200'
                        : darkMode
                          ? 'bg-blue-900/20 border-blue-600'
                          : 'bg-blue-50 border-blue-400'
                  }`}>
                    {msg.result.type === 'bias' && <AlertCircle className={`w-5 h-5 mr-2 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`} />}
                    {msg.result.type === 'neutral' && <CheckCircle className={`w-5 h-5 mr-2 ${darkMode ? 'text-slate-400' : 'text-gray-500'}`} />}
                    {msg.result.type === 'success' && <CheckCircle className={`w-5 h-5 mr-2 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />}
                    
                    <div className="flex-1">
                      <p className="text-sm font-medium">{msg.result.content}</p>
                    </div>
                  </div>
                )}
                
                {msg.preview && (
                  <div className={`mt-3 p-3 rounded border text-sm ${
                    darkMode ? 'bg-slate-800 border-slate-600' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <p className={`font-semibold mb-1 ${darkMode ? 'text-slate-300' : 'text-gray-700'}`}>
                      Preview:
                    </p>
                    <p className={darkMode ? 'text-slate-400' : 'text-gray-600'}>{msg.preview}</p>
                  </div>
                )}
                
                {msg.options && msg.options.length > 0 && (
                  <div className="mt-3 space-y-2">
                    {msg.options.map((option) => (
                      <button type="button"
                        key={option.id}
                        onClick={() => handleOptionClick(option.id, option.text)}
                        onKeyDown={(e) => handleKeyPress(e, () => handleOptionClick(option.id, option.text))}
                        disabled={loading || batchProcessing}
                        className={`w-full text-left px-4 py-2 rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 ${
                          loading || batchProcessing
                            ? 'opacity-50 cursor-not-allowed'
                            : darkMode
                              ? 'bg-slate-600 hover:bg-slate-500 text-white focus:ring-blue-400'
                              : 'bg-blue-100 hover:bg-blue-200 text-blue-900 focus:ring-blue-500'
                        }`}
                        tabIndex={0}
                      >
                        {option.text}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Batch Processing Results */}
          {batchProcessing && (
            <div className={`rounded-lg p-4 ${darkMode ? 'bg-slate-700' : 'bg-white shadow-md'}`}>
              <div className="flex items-center justify-between mb-3">
                <h3 className={`font-semibold flex items-center ${darkMode ? 'text-slate-200' : 'text-gray-900'}`}>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Analyzing document...
                </h3>
                <span className={`text-sm ${darkMode ? 'text-slate-400' : 'text-gray-600'}`}>
                  {batchProgress.current} / {batchProgress.total}
                </span>
              </div>
              
              <div className={`w-full h-2 rounded-full mb-4 ${darkMode ? 'bg-slate-600' : 'bg-gray-200'}`}>
                <div
                  className="h-full bg-blue-600 rounded-full transition-all duration-300"
                  style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
                />
              </div>



              <div className="space-y-2 max-h-64 overflow-y-auto">
                {allResults.map((result, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded border-l-4 ${
                      result.error
                        ? darkMode
                          ? 'bg-red-900/20 border-red-600'
                          : 'bg-red-50 border-red-400'
                        : result.result.includes('Label: Bias')
                          ? darkMode
                            ? 'bg-yellow-900/20 border-yellow-600'
                            : 'bg-yellow-50 border-yellow-400'
                          : darkMode
                            ? 'bg-blue-900/20 border-blue-600'
                            : 'bg-blue-50 border-blue-400'
                    }`}
                  >

                    
                    <div className="flex items-start">

                      
                      {result.error ? (
                        <AlertCircle className={`w-5 h-5 mr-2 flex-shrink-0 ${darkMode ? 'text-red-400' : 'text-red-600'}`} />
                      ) : result.result.includes('Label: Bias') ? (
                        <AlertCircle className={`w-5 h-5 mr-2 flex-shrink-0 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`} />
                      ) : (
                        <CheckCircle className={`w-5 h-5 mr-2 flex-shrink-0 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                      )}
                      <div className="flex-1">

                        
                        <p className={`text-sm font-semibold ${darkMode ? 'text-slate-200' : 'text-gray-900'}`}>
                          Chunk {result.chunk_number}
                        </p>
                        <p className={`text-xs mt-1 ${darkMode ? 'text-slate-400' : 'text-gray-600'}`}>
                          {result.content_preview}
                        </p>

                        <p className="text-xs italic opacity-75 mt-2">
                        Analyzed content:
                        </p>

                        <p className="text-xs whitespace-pre-wrap mt-1">
                        
                        </p>

                        <p className={`text-sm mt-2 font-medium ${darkMode ? 'text-slate-300' : 'text-gray-800'}`}>
                          {result.result}
                        </p>
                      </div>
                    </div>
                  </div>

                  
                ))}
              </div>
            </div>
          )}

          {loading && !batchProcessing && (
            <div className="flex justify-start">
              <div className={`rounded-lg p-4 flex items-center ${
                darkMode ? 'bg-slate-700' : 'bg-white shadow-md'
              }`}>
                <Loader2 className={`w-5 h-5 mr-3 animate-spin ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                <span className={darkMode ? 'text-slate-300' : 'text-gray-700'}>Processing...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className={`p-4 border-t transition-colors duration-300 ${
          darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-gray-200'
        }`}>
          {textInputEnabled && (
            <div>
              <div className="flex gap-2">
                <textarea
                  rows={4}
                  value={textInput}
                  onChange={(e) => {
                    const value = e.target.value;
                    const wordCount = value.trim().split(/\s+/).filter(Boolean).length;

                    if (wordCount > MAX_WORDS) {
                      setTextError(`Maximum ${MAX_WORDS} words allowed. Currently: ${wordCount}`);
                    } else {
                      setTextError(null);
                    }

                    setTextInput(value);
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && handleTextSubmit()}
                  placeholder="Type your text here..."
                  disabled={loading}
                  className={`resize-none flex-1 px-4 py-3 rounded-lg transition-colors duration-300 
                    focus:outline-none focus:ring-2 border
                    ${
                      textError
                        ? 'border-2 border-red-500 focus:ring-red-500'
                        : darkMode
                          ? 'bg-slate-700 border-slate-600 text-slate-100 placeholder-slate-400 focus:ring-blue-500'
                          : 'bg-gray-50 border-gray-300 text-gray-900 placeholder-gray-500 focus:ring-blue-400'
                    }
                  `}
                />

                <button
                  type="button"
                  onClick={handleTextSubmit}
                  disabled={!textInput.trim() || loading || textError !== null}
                  className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 flex items-center ${
                    textInput.trim() && !loading && !textError
                      ? darkMode
                        ? 'bg-blue-700 hover:bg-blue-600 text-white'
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>

              {/* Word Counter */}
              <p
                className={`text-xs mt-2 ${
                  textInput.trim().split(/\s+/).filter(Boolean).length > MAX_WORDS
                    ? 'text-red-500'
                    : 'text-gray-500'
                }`}
              >
                {textInput.trim().split(/\s+/).filter(Boolean).length} / {MAX_WORDS} words
              </p>

              {/* Error Message */}
              {textError && (
                <p className="text-red-500 text-sm mt-1">
                  {textError}
                </p>
              )}
            </div>
          )}


          {fileUploadEnabled && (
            <div className="text-center">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={loading}
                className="hidden"
                id="file-input"
              />
              <label
                htmlFor="file-input"
                className={`inline-flex items-center px-6 py-3 rounded-lg font-medium cursor-pointer transition-all duration-300 ${
                  loading
                    ? 'opacity-50 cursor-not-allowed'
                    : darkMode
                      ? 'bg-blue-700 hover:bg-blue-600 text-white'
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                <Upload className="w-5 h-5 mr-2" />
                Choose PDF File
              </label>
            </div>
          )}

          {!textInputEnabled && !fileUploadEnabled && !loading && (
            <p className={`text-center text-sm ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
              Select an option above to continue
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
