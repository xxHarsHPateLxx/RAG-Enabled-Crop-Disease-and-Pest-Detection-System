import React, { useState, useRef } from "react";
import { Upload, Loader2, AlertCircle, ArrowLeft, Camera } from "lucide-react";
import { useNavigate } from "react-router-dom";

const SESSION_STORAGE_KEY = "agrisentry_session_id";

const getSessionId = () => {
  const storedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (storedSessionId) {
    return storedSessionId;
  }

  const generatedSessionId = window.crypto.randomUUID();
  window.localStorage.setItem(SESSION_STORAGE_KEY, generatedSessionId);
  return generatedSessionId;
};

export default function ImageInput() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [predictionMode, setPredictionMode] = useState("multimodal_llm");
  const [cropName, setCropName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!imageFile) {
      setError("Please select an image.");
      return;
    }

    if (predictionMode === "cnn_llm" && !cropName) {
      setError("Please select a crop for CNN + LLM mode.");
      return;
    }

    setError(null);
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("prediction_mode", predictionMode);
      if (predictionMode === "cnn_llm") {
        formData.append("crop", cropName);
      }
      formData.append("session_id", getSessionId());

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      if (data.session_id) {
        window.localStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
      }

      console.log(data);
      navigate("/results", { state: { image: imagePreview, result: data } });
    } catch (err) {
      setError("Failed to process image. Try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Back Button */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 mb-6 font-medium"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Home</span>
        </button>

        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Plant Disease Detection
          </h1>
          <p className="text-lg text-gray-600">
            Choose a prediction mode and upload or capture a plant image
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-gray-800">Prediction mode</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setPredictionMode("multimodal_llm")}
                className={`rounded-xl border p-4 text-left transition ${predictionMode === "multimodal_llm" ? "border-emerald-500 bg-emerald-50" : "border-gray-200 bg-white hover:border-emerald-300"}`}
              >
                <p className="font-semibold text-gray-900">Multimodal LLM</p>
                <p className="text-sm text-gray-600 mt-1">No crop selection. The model infers both crop and disease from image.</p>
              </button>

              <button
                type="button"
                onClick={() => setPredictionMode("cnn_llm")}
                className={`rounded-xl border p-4 text-left transition ${predictionMode === "cnn_llm" ? "border-blue-500 bg-blue-50" : "border-gray-200 bg-white hover:border-blue-300"}`}
              >
                <p className="font-semibold text-gray-900">CNN + LLM</p>
                <p className="text-sm text-gray-600 mt-1">Select crop manually. CNN predicts disease, LLM generates recommendations.</p>
              </button>
            </div>
          </div>

          {predictionMode === "cnn_llm" && (
            <select
              value={cropName}
              onChange={(e) => setCropName(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 text-gray-700"
            >
              <option value="">Select Crop</option>
              <option value="Wheat">Wheat</option>
              <option value="Rice">Rice</option>
              <option value="Maize">Maize</option>
            </select>
          )}

          {/* File Upload */}
          <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-blue-300 rounded-xl hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 cursor-pointer">
            <Upload className="w-16 h-16 text-blue-500 mb-4" />
            <h3 className="text-xl font-semibold text-gray-800 mb-2">
              Upload Image
            </h3>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* Camera Option (Mobile only) */}
          {isMobile && (
            <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-green-300 rounded-xl hover:border-green-400 hover:bg-green-50 transition-all duration-200 cursor-pointer">
              <Camera className="w-16 h-16 text-green-500 mb-4" />
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                Use Camera
              </h3>
              <input
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          )}

          {/* Preview */}
          {imagePreview && (
            <div className="flex justify-center">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-64 h-auto rounded-lg border border-gray-300"
              />
            </div>
          )}

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition disabled:bg-gray-400"
          >

            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin" /> Processing...
              </span>
            ) : (
              "Upload & Predict"
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
