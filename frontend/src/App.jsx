import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './index.css';

const CharcoalTheme = {
    background: '#121212',
    surface: '#1e1e1e',
    surfaceHover: '#2a2a2a',
    primary: '#f5f5f5',
    secondary: '#a0a0a0',
    accent: '#2563eb', // A slight blue accent for success/actions
    error: '#cf6679',
};

function Scene() {
    return (
        <Canvas
            style={{ position: 'absolute', top: 0, left: 0, zIndex: -1, width: '100vw', height: '100vh', background: CharcoalTheme.background }}
            camera={{ position: [0, 0, 1] }}
        >
            <ambientLight intensity={0.5} />
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            <OrbitControls autoRotate autoRotateSpeed={0.5} enableZoom={false} />
        </Canvas>
    );
}

function App() {
    const [option, setOption] = useState(null);
    const [textQuery, setTextQuery] = useState('');
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const options = ['Text', 'Audio', 'Image', 'Video'];

    const resetState = () => {
        setTextQuery('');
        setFile(null);
        setResult(null);
        setError(null);
    };

    const handleOptionSelect = (opt) => {
        setOption(opt);
        resetState();
    };

    const checkText = async () => {
        if (!textQuery.trim()) {
            setError('Please enter some text before checking');
            return;
        }
        setLoading(true);
        setResult(null);
        setError(null);
        try {
            const res = await axios.post('http://localhost:8000/api/check-text', { query: textQuery });
            setResult(res.data.result);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Error occurred');
        }
        setLoading(false);
    };

    const checkMedia = async (endpoint) => {
        if (!file) {
            setError('Please upload a file before checking');
            return;
        }
        setLoading(true);
        setResult(null);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await axios.post(`http://localhost:8000/api/${endpoint}`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(res.data.result);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Error occurred');
        }
        setLoading(false);
    };

    return (
        <>
            <Scene />
            <div className="app-container">
                <motion.h1
                    initial={{ y: -50, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.8 }}
                    className="main-title"
                >
                    Deepfake Detection
                </motion.h1>

                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                    className="content-box"
                    style={{
                        background: CharcoalTheme.surface,
                        border: `1px solid ${CharcoalTheme.surfaceHover}`
                    }}
                >
                    <div className="options-container">
                        <p className="options-subtitle">Choose the type of data you want to check:</p>
                        <div className="options-grid">
                            {options.map(opt => (
                                <button
                                    key={opt}
                                    onClick={() => handleOptionSelect(opt)}
                                    className="option-button"
                                    style={{
                                        background: option === opt ? CharcoalTheme.primary : 'transparent',
                                        color: option === opt ? CharcoalTheme.background : CharcoalTheme.primary,
                                        border: `2px solid ${CharcoalTheme.primary}`,
                                        fontWeight: option === opt ? 'bold' : 'normal',
                                    }}
                                >
                                    {opt}
                                </button>
                            ))}
                        </div>
                    </div>

                    <AnimatePresence mode="wait">
                        {option && (
                            <motion.div
                                key={option}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ duration: 0.4 }}
                            >
                                <h2 className="section-title" style={{ borderBottom: `1px solid ${CharcoalTheme.surfaceHover}` }}>
                                    {option === 'Text' ? 'Text News Verification' :
                                        option === 'Audio' ? 'Audio News Verification' :
                                            option === 'Image' ? 'Image Deepfake Detection' : 'Video Deepfake Detection'}
                                </h2>

                                <div className="input-group">
                                    {option === 'Text' ? (
                                        <textarea
                                            rows={5}
                                            className="text-input"
                                            placeholder="Enter the news statement or article..."
                                            value={textQuery}
                                            onChange={(e) => setTextQuery(e.target.value)}
                                            style={{
                                                background: CharcoalTheme.background,
                                                border: `1px solid ${CharcoalTheme.surfaceHover}`,
                                                color: CharcoalTheme.primary,
                                            }}
                                        />
                                    ) : (
                                        <div className="file-drop-area" style={{
                                            border: `2px dashed ${CharcoalTheme.surfaceHover}`,
                                            background: CharcoalTheme.background
                                        }}>
                                            <label style={{ cursor: 'pointer', display: 'block', width: '100%', height: '100%' }}>
                                                <span>Click to upload {option} file</span>
                                                <input
                                                    type="file"
                                                    accept={
                                                        option === 'Audio' ? 'audio/mp3,audio/wav,audio/m4a' :
                                                            option === 'Image' ? 'image/jpeg,image/png,image/jpg' :
                                                                'video/mp4,video/avi,video/quicktime,video/x-matroska'
                                                    }
                                                    onChange={(e) => setFile(e.target.files[0])}
                                                    style={{ display: 'none' }}
                                                />
                                            </label>
                                            {file && <p style={{ marginTop: '1rem', color: CharcoalTheme.secondary }}>Selected: {file.name}</p>}
                                        </div>
                                    )}

                                    <button
                                        className="submit-button"
                                        onClick={() => {
                                            if (option === 'Text') checkText();
                                            else checkMedia(`check-${option.toLowerCase()}`);
                                        }}
                                        disabled={loading}
                                        style={{
                                            background: CharcoalTheme.primary,
                                            color: CharcoalTheme.background,
                                            opacity: loading ? 0.7 : 1,
                                            cursor: loading ? 'not-allowed' : 'pointer',
                                        }}
                                    >
                                        {loading ? 'Analyzing...' : 'Check'}
                                    </button>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    <AnimatePresence>
                        {error && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="message-box"
                                style={{
                                    color: CharcoalTheme.error,
                                    background: 'rgba(207, 102, 121, 0.1)',
                                    border: `1px solid ${CharcoalTheme.error}`
                                }}
                            >
                                Error: {error}
                            </motion.div>
                        )}

                        {result !== null && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="message-box"
                                style={{
                                    color: CharcoalTheme.primary,
                                    background: 'rgba(37, 99, 235, 0.1)',
                                    border: `1px solid ${CharcoalTheme.accent}`
                                }}
                            >
                                <strong>Result: </strong> {typeof result === 'object' ? JSON.stringify(result) : result}
                            </motion.div>
                        )}
                    </AnimatePresence>

                </motion.div>
            </div>
        </>
    );
}

export default App;
