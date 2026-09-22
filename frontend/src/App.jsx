import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);

  const modes = [
    {
      id: "image",
      title: "Image",
      icon: "🖼️",
      description: "Detect manipulated faces in images",
    },
    {
      id: "video",
      title: "Video",
      icon: "🎥",
      description: "Analyze video frames for deepfakes",
    },
    {
      id: "audio",
      title: "Audio",
      icon: "🎙️",
      description: "Detect synthetic or cloned voices",
    },
    {
      id: "audiovisual",
      title: "Audio + Video",
      icon: "🎬",
      description: "Combine visual and audio evidence",
    },
  ];

  const allowedTypes = {
    image: "image/*",
    video: "video/*",
    audio: "audio/*",
    audiovisual: "video/*",
  };

  const modeNames = {
    image: "image",
    video: "video",
    audio: "audio",
    audiovisual: "video",
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setFile(null);
    setResult(null);
    setDragActive(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const validateFile = (selectedFile) => {
    if (!selectedFile) {
      return false;
    }

    if (mode === "image") {
      return selectedFile.type.startsWith("image/");
    }

    if (mode === "video" || mode === "audiovisual") {
      return selectedFile.type.startsWith("video/");
    }

    if (mode === "audio") {
      return selectedFile.type.startsWith("audio/");
    }

    return false;
  };

  const selectFile = (selectedFile) => {
    if (!validateFile(selectedFile)) {
      setResult({
        error: `Please select a valid ${modeNames[mode]} file.`,
      });

      setFile(null);
      return;
    }

    setFile(selectedFile);
    setResult(null);
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (selectedFile) {
      selectFile(selectedFile);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);

    const droppedFile = event.dataTransfer.files[0];

    if (droppedFile) {
      selectFile(droppedFile);
    }
  };

  const handleChooseFile = () => {
    fileInputRef.current?.click();
  };

  const removeFile = () => {
    setFile(null);
    setResult(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";

    const units = ["Bytes", "KB", "MB", "GB"];
    const index = Math.floor(
      Math.log(bytes) / Math.log(1024)
    );

    return `${(bytes / Math.pow(1024, index)).toFixed(2)} ${
      units[index]
    }`;
  };

 const handleDetect = async () => {
  if (!file) {
    setResult({ error: "Please select a file first." });
    return;
  }

  setLoading(true);
  setResult(null);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(
      `https://multimodal-deepfake-detector-q8kv.onrender.com/predict/${mode}`,
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || "Detection failed."
      );
    }

    setResult({
      prediction: data.prediction,
      confidence: Number(data.confidence)
    });

  } catch (error) {
    setResult({
      error:
        error.message ||
        "Could not connect to the detection server."
    });
  } finally {
    setLoading(false);
  }
};

  return (
    <div className="app">

      <header className="header">

        <div className="logo">
          🛡️
        </div>

        <div>
          <h1>
            Multimodal Deepfake Detector
          </h1>

          <p>
            AI-powered detection using visual, audio,
            and audio-visual analysis
          </p>
        </div>

      </header>

      <main className="container">

        <section className="hero">

          <div className="hero-badge">
            ● AI-POWERED MEDIA FORENSICS
          </div>

          <h2>
            Detect Deepfakes
            <br />
            with Multiple AI Models
          </h2>

          <p>
            Analyze images, videos, audio, and combined
            audio-visual content using deep learning.
          </p>

        </section>

        <section className="mode-grid">

          {modes.map((item) => (

            <button
              key={item.id}
              className={`mode-card ${
                mode === item.id ? "active" : ""
              }`}
              onClick={() =>
                handleModeChange(item.id)
              }
            >

              <span className="mode-icon">
                {item.icon}
              </span>

              <span className="mode-title">
                {item.title}
              </span>

              <span className="mode-description">
                {item.description}
              </span>

              {mode === item.id && (
                <span className="active-indicator">
                  ✓
                </span>
              )}

            </button>

          ))}

        </section>

        <section className="upload-section">

          <div
            className={`upload-box ${
              dragActive ? "drag-active" : ""
            } ${
              file ? "has-file" : ""
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={!file ? handleChooseFile : undefined}
          >

            {!file ? (

              <>
                <div className="upload-icon">
                  ⬆
                </div>

                <h3>
                  Drag & drop your {mode === "audiovisual"
                    ? "video"
                    : mode} here
                </h3>

                <p>
                  or click to browse files from your computer
                </p>

                <span className="supported-text">
                  Supported format: {mode === "image"
                    ? "JPG, PNG, WEBP"
                    : mode === "video" || mode === "audiovisual"
                    ? "MP4, AVI, MOV, MKV"
                    : "WAV, MP3, FLAC"}
                </span>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept={allowedTypes[mode]}
                  onChange={handleFileChange}
                />

                <button
                  type="button"
                  className="choose-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleChooseFile();
                  }}
                >
                  Choose File
                </button>
              </>

            ) : (

              <div className="file-preview">

                <div className="file-icon">
                  {mode === "image"
                    ? "🖼️"
                    : mode === "audio"
                    ? "🎙️"
                    : "🎥"}
                </div>

                <div className="file-info">

                  <span className="file-ready">
                    ✓ File ready for analysis
                  </span>

                  <h3>
                    {file.name}
                  </h3>

                  <p>
                    {formatFileSize(file.size)}
                  </p>

                </div>

                <button
                  type="button"
                  className="remove-button"
                  onClick={removeFile}
                >
                  ✕
                </button>

              </div>

            )}

          </div>

          <button
            className="detect-button"
            onClick={handleDetect}
            disabled={loading || !file}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing Media...
              </>
            ) : (
              <>
                🔍 Detect Deepfake
              </>
            )}
          </button>

        </section>

        {result && (

          <section className="result-section">

            <h2>
              Detection Result
            </h2>

            {result.error ? (

              <div className="error-result">
                <strong>
                  Detection Error
                </strong>
                <span>
                  {result.error}
                </span>
              </div>

            ) : (

              <div className="result-card">

                <div className="result-label">
                  Analysis Complete
                </div>

                <div
                  className={`prediction ${
                    result.prediction?.toLowerCase()
                  }`}
                >
                  {result.prediction}
                </div>

                <div className="confidence">
                  Confidence:
                  <strong>
                    {" "}
                    {(result.confidence).toFixed(2)}%
                  </strong>
                </div>

              </div>

            )}

          </section>

        )}

      </main>

      <footer>

        <p>
          Advanced Multimodal Deepfake Detection
        </p>

        <p>
          Visual • Audio • Audio-Visual Feature Fusion
        </p>

      </footer>

    </div>
  );
}

export default App;