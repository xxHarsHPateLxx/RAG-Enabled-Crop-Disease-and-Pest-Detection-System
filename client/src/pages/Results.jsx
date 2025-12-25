import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertTriangle, Leaf, CheckCircle } from 'lucide-react';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const image = location.state?.image;
  const result = location.state?.result;

  const { crop, disease, confidence, advice } = result;

  // Enhanced function to convert markdown-style text to JSX with proper formatting
  const renderMarkdownText = (text) => {
    if (!text) return null;
    
    const elements = [];
    let currentIndex = 0;
    
    // Split text into lines for processing
    const lines = text.split('\n');
    let i = 0;
    
    while (i < lines.length) {
      let line = lines[i];
      
      // Skip empty lines
      if (!line.trim()) {
        i++;
        continue;
      }
      
      // Check for h1 headings (# text or **text** at start)
      const h1Match = line.match(/^#\s+\*\*(.+?)\*\*\s*$/);
      if (h1Match) {
        elements.push(
          <h2 key={`h1-${currentIndex++}`} className="text-2xl font-bold text-green-800 mt-8 mb-4 first:mt-0 border-b-2 border-green-200 pb-2">
            {h1Match[1]}
          </h2>
        );
        i++;
        continue;
      }
      
      // Check for h2 headings (## **text**)
      const h2Match = line.match(/^##\s+\*\*(.+?)\*\*\s*$/);
      if (h2Match) {
        elements.push(
          <h3 key={`h2-${currentIndex++}`} className="text-xl font-bold text-green-700 mt-6 mb-3 first:mt-0">
            {h2Match[1]}
          </h3>
        );
        i++;
        continue;
      }
      
      // Check for standalone bold headings (**text**)
      const headingMatch = line.match(/^\*\*(.+?)\*\*:?\s*$/);
      if (headingMatch) {
        elements.push(
          <h4 key={`heading-${currentIndex++}`} className="text-lg font-semibold text-green-700 mt-5 mb-2 first:mt-0">
            {headingMatch[1]}
          </h4>
        );
        i++;
        continue;
      }
      
      // Check for bullet points (- text or • text)
      if (line.trim().startsWith('-') || line.trim().startsWith('•')) {
        const bulletItems = [];
        while (i < lines.length && (lines[i].trim().startsWith('-') || lines[i].trim().startsWith('•'))) {
          let bulletText = lines[i].trim().substring(1).trim();
          // Remove markdown formatting from bullet text
          bulletText = bulletText.replace(/\*\*(.+?)\*\*/g, '$1');
          
          // Check if this bullet point ends with a colon (it's a subheading)
          if (bulletText.endsWith(':')) {
            // Render any accumulated bullet items
            if (bulletItems.length > 0) {
              elements.push(
                <ul key={`list-${currentIndex++}`} className="list-disc list-inside ml-4 mb-4 space-y-2">
                  {bulletItems}
                </ul>
              );
              bulletItems.length = 0; // Clear the array
            }
            
            // Render the subheading
            elements.push(
              <h5 key={`subheading-${currentIndex++}`} className="text-base font-semibold text-green-600 mt-4 mb-2">
                {bulletText}
              </h5>
            );
            i++;
          } else {
            bulletItems.push(
              <li key={`bullet-${currentIndex++}`} className="mb-2 text-gray-700">
                {bulletText}
              </li>
            );
            i++;
          }
        }
        
        // Render any remaining bullet items
        if (bulletItems.length > 0) {
          elements.push(
            <ul key={`list-${currentIndex++}`} className="list-disc list-inside ml-4 mb-4 space-y-2">
              {bulletItems}
            </ul>
          );
        }
        continue;
      }
      
      // Check for numbered lists (1. text)
      if (line.trim().match(/^\d+\./)) {
        const numberedItems = [];
        while (i < lines.length && lines[i].trim().match(/^\d+\./)) {
          let numberText = lines[i].trim().replace(/^\d+\./, '').trim();
          // Remove markdown formatting from numbered text
          numberText = numberText.replace(/\*\*(.+?)\*\*/g, '$1');
          numberedItems.push(
            <li key={`number-${currentIndex++}`} className="mb-2 text-gray-700">
              {numberText}
            </li>
          );
          i++;
        }
        elements.push(
          <ol key={`numlist-${currentIndex++}`} className="list-decimal list-inside ml-4 mb-4 space-y-2">
            {numberedItems}
          </ol>
        );
        continue;
      }
      
      // Regular paragraph text - remove any remaining markdown
      if (line.trim()) {
        let cleanText = line.trim();
        // Remove all markdown formatting
        cleanText = cleanText.replace(/^#+\s+/g, ''); // Remove # headers
        cleanText = cleanText.replace(/\*\*(.+?)\*\*/g, '$1'); // Remove bold markers
        
        if (cleanText) {
          elements.push(
            <p key={`para-${currentIndex++}`} className="text-gray-700 leading-relaxed mb-3">
              {cleanText}
            </p>
          );
        }
      }
      
      i++;
    }
    
    return elements;
  };

  const handleTryAnother = () => {
    navigate('/');
  };

  const handleBack = () => {
    navigate(-1);
  };

  if (!image) {
    navigate('/');
    return null;
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">

        <button
          onClick={handleBack}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 mb-6 font-medium"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back</span>
        </button>

        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">

          <div className="bg-gradient-to-r from-green-600 to-green-700 px-8 py-6">
            <h1 className="text-3xl font-bold text-white text-center">
              Analysis Results
            </h1>
            <p className="text-green-100 text-center mt-2">
              AI-powered crop disease detection complete
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-8">
            {/* Image Section */}
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-green-800">Analyzed Image</h2>
              <img
                src={image}
                alt="Analyzed crop"
                className="w-full h-80 object-cover rounded-xl shadow-lg"
              />
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              <div className="bg-green-50 p-6 rounded-xl">
                <div className="flex items-center space-x-3 mb-4">
                  <Leaf className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-green-800">Crop</h3>
                </div>
                <p className="text-2xl font-bold text-green-700"> {crop}</p>
                {/* <p className="text-green-600 mt-1">Confidence: 100%</p> */}
              </div>

              <div className="bg-red-50 p-6 rounded-xl border border-red-200">
                <div className="flex items-center space-x-3 mb-4">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
                  <h3 className="text-lg font-semibold text-red-800">Detected Issue</h3>
                </div>
                <p className="text-2xl font-bold text-red-700"> {disease}</p>
                <p className="text-red-600 mt-1">Confidence: {confidence*100}%</p>
              </div>

            </div>

          </div>

          {/* AI-Generated Advice Section */}
          <div className="px-8 pb-8">
            <div className="bg-gradient-to-br from-blue-50 to-green-50 p-8 rounded-xl border border-green-200 shadow-inner">
              <div className="flex items-center space-x-3 mb-6">
                <CheckCircle className="h-7 w-7 text-green-600" />
                <h2 className="text-2xl font-bold text-gray-800">AI-Powered Analysis & Recommendations</h2>
              </div>
              
              <div className="prose prose-lg max-w-none">
                {renderMarkdownText(advice)}
              </div>
            </div>
          </div>

          <div className="bg-gray-50 px-8 py-6 text-center">
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