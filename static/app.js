import { AutoTokenizer, MusicgenForConditionalGeneration, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

// Global Configuration
env.allowLocalModels = false;

// App State
let audioCtx = null;
let tokenizer = null;
let model = null;
let isAiLoading = false;
let isGenerating = false;
let isPlaying = false;
let isLooping = false;
let playheadPosition = 0; // Current time in seconds
let playStartTime = 0; // AudioContext currentTime when playback started
let pauseOffset = 0; // Time in seconds where we paused
let animationFrameId = null;

// Timeline parameters (scaled dynamically by BPM)
let secondsPerBeat = 0.5; // 120 BPM default
let secondsPerBar = 2.0; // 4/4 time
let totalDuration = 80.0; // 16 clip-slots × 5s = 80s
const pixelsPerSecond = 20; // Scale: 1 second = 20px, 5s clip = 100px
const clipDuration = 5.0; // Target duration of loop in seconds
const headerWidth = 180; // Width of the track-info panel in px (must match --header-width CSS var)
const numTracks = 4;
const minClipDuration = 0.25;

// Audio routing nodes
let trackVolumeNodes = [];
let trackMuteStates = [false, false, false, false];
let trackSoloStates = [false, false, false, false];
let masterGainNode = null;

// Track settings
const trackColors = ['#00f2fe', '#bd00ff', '#ff007f', '#ff9f00'];

// Data Stores
const libraryClips = []; // List of generated sound clips
const placedClips = []; // Clips placed in the DAW: { id, clipId, trackId, startTime, duration }
let activeAudioSources = []; // Currently playing AudioBufferSourceNodes
let activeClipGesture = null; // Current move/resize interaction state

// UI Elements
const elBtnPlay = document.getElementById('btn-play');
const elBtnPause = document.getElementById('btn-pause');
const elBtnStop = document.getElementById('btn-stop');
const elBtnLoop = document.getElementById('btn-loop');
const elBpmInput = document.getElementById('bpm-input');
const elSnapSelect = document.getElementById('snap-select');
const elBtnExport = document.getElementById('btn-export');
const elBtnHelp = document.getElementById('btn-help');
const elBtnCloseHelp = document.getElementById('btn-close-help');
const elBtnGetStarted = document.getElementById('btn-get-started');
const elHelpModal = document.getElementById('help-modal');
const elEngineStatus = document.getElementById('engine-status');
const elStatusText = document.getElementById('status-text');
const elProgressContainer = document.getElementById('model-progress-container');
const elProgressStatus = document.getElementById('progress-status');
const elProgressPercent = document.getElementById('progress-percent');
const elProgressBar = document.getElementById('model-progress-bar');
const elLibraryEmpty = document.getElementById('library-empty');
const elLibraryList = document.getElementById('library-list');
const elLibraryCount = document.getElementById('library-count');
const elRuler = document.getElementById('timeline-ruler');
const elPlayhead = document.getElementById('playhead');
const elTrackLanes = document.querySelectorAll('.track-lane');

// Structured inputs
const elInputInstrument = document.getElementById('input-instrument');
const elInputStyle = document.getElementById('input-style');
const elInputMood = document.getElementById('input-mood');
const elInputBpm = document.getElementById('input-bpm');
const elSyncBpmBtn = document.getElementById('sync-bpm-btn');
const elBtnGenerate = document.getElementById('btn-generate');
const elBtnGenerateText = document.getElementById('btn-generate-text');

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    setupUIEventListeners();
    updateTempoSettings(); // Calculates initial duration and renders ruler
    setupAudioContext();
    loadAIModel();
    
    // Show help modal on startup
    elHelpModal.classList.remove('hidden');
});

// Setup Web Audio API Context
function setupAudioContext() {
    window.addEventListener('click', ensureAudioContext);
    window.addEventListener('keydown', ensureAudioContext);
    window.addEventListener('dragstart', ensureAudioContext);
}

function ensureAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        masterGainNode = audioCtx.createGain();
        masterGainNode.gain.value = 0.95; // Limit headroom
        masterGainNode.connect(audioCtx.destination);

        // Create gain nodes for each of the 4 tracks
        for (let i = 0; i < numTracks; i++) {
            const gainNode = audioCtx.createGain();
            gainNode.gain.value = 0.8; // Default volume
            gainNode.connect(masterGainNode);
            trackVolumeNodes.push(gainNode);
        }

        window.removeEventListener('click', ensureAudioContext);
        window.removeEventListener('keydown', ensureAudioContext);
        window.removeEventListener('dragstart', ensureAudioContext);
        console.log("AudioContext initialized.");
    }

    return audioCtx;
}

// Update settings when Project BPM changes
function updateTempoSettings() {
    const bpm = parseFloat(elBpmInput.value) || 120;
    secondsPerBeat = 60 / bpm;
    secondsPerBar = secondsPerBeat * 4;
    // Timeline: always exactly 16 clip-slots wide (16 × 5s = 80s)
    // This keeps the ruler aligned to clips regardless of BPM.
    totalDuration = 16 * clipDuration; // 80 seconds
    
    // Re-render ruler
    renderTimelineRuler();
    
    // Update grid spacing: minor lines every beat, major lines every clipDuration (5s)
    const beatWidth = secondsPerBeat * pixelsPerSecond;
    const clipWidth = clipDuration * pixelsPerSecond; // 5s = 100px
    
    elTrackLanes.forEach(lane => {
        lane.style.backgroundSize = `${beatWidth}px 100%, ${clipWidth}px 100%`;
    });
    
    // Adjust playhead position visually to match new scale (with header offset)
    elPlayhead.style.left = `${headerWidth + playheadPosition * pixelsPerSecond}px`;
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function getSnapUnitSeconds() {
    const snapMode = elSnapSelect.value;

    if (snapMode === 'beat') return secondsPerBeat;
    if (snapMode === 'bar') return secondsPerBar;
    return 0;
}

function snapTimeToGrid(timeInSeconds) {
    const snapUnit = getSnapUnitSeconds();
    if (!snapUnit) return timeInSeconds;
    return Math.round(timeInSeconds / snapUnit) * snapUnit;
}

function normalizePlacementTime(startTime, duration = clipDuration) {
    const maxStartTime = Math.max(0, totalDuration - duration);
    const clamped = clamp(startTime, 0, maxStartTime);
    return clamp(snapTimeToGrid(clamped), 0, maxStartTime);
}

function getTrackLaneById(trackId) {
    return document.querySelector(`.track-lane[data-track-id="${trackId}"]`);
}

function getTrackIdFromClientY(clientY, fallbackTrackId) {
    const row = Array.from(document.querySelectorAll('.daw-track-row')).find(trackRow => {
        const rect = trackRow.getBoundingClientRect();
        return clientY >= rect.top && clientY <= rect.bottom;
    });

    if (!row) return fallbackTrackId;

    const lane = row.querySelector('.track-lane');
    if (!lane) return fallbackTrackId;

    const trackId = parseInt(lane.getAttribute('data-track-id'), 10);
    return Number.isFinite(trackId) ? trackId : fallbackTrackId;
}

function getPlacementById(placedId) {
    return placedClips.find(pc => pc.id === placedId) || null;
}

function getClipByPlacement(placement) {
    return placement ? libraryClips.find(c => c.id === placement.clipId) || null : null;
}

function updatePlacedClipElement(placement, clipBox) {
    clipBox.style.left = `${placement.startTime * pixelsPerSecond}px`;
    clipBox.style.width = `${placement.duration * pixelsPerSecond}px`;
    clipBox.setAttribute('data-track-id', placement.trackId);
}

function redrawPlacedClipWaveform(placement) {
    const clip = getClipByPlacement(placement);
    const clipBox = document.getElementById(placement.id);
    if (!clip || !clipBox) return;

    const canvas = clipBox.querySelector('.placed-clip-waveform');
    if (!canvas) return;

    drawWaveform(clip.audioBuffer, canvas, trackColors[placement.trackId], {
        repeatDuration: placement.duration,
    });
}

function beginPlacedClipGesture(e, placementId, mode, resizeSide = null) {
    const placement = getPlacementById(placementId);
    if (!placement) return;

    const clip = getClipByPlacement(placement);
    if (!clip) return;

    const clipBox = document.getElementById(placementId);
    if (!clipBox) return;

    e.preventDefault();
    e.stopPropagation();

    const clipRect = clipBox.getBoundingClientRect();
    const pointerOffsetX = clamp(e.clientX - clipRect.left, 0, clipRect.width);

    activeClipGesture = {
        pointerId: e.pointerId,
        placementId,
        clipId: clip.id,
        mode,
        resizeSide,
        pointerOffsetX,
        initialTrackId: placement.trackId,
        initialStartTime: placement.startTime,
        initialDuration: placement.duration,
        clipBox,
        clip,
    };

    clipBox.classList.add(mode === 'move' ? 'dragging' : 'resizing');
}

function updatePlacedClipGesture(e) {
    if (!activeClipGesture || e.pointerId !== activeClipGesture.pointerId) return;

    const { placementId, clipBox, mode, resizeSide, pointerOffsetX, initialStartTime, initialDuration, initialTrackId } = activeClipGesture;
    const placement = getPlacementById(placementId);
    if (!placement || !clipBox) return;

    if (mode === 'move') {
        const targetTrackId = getTrackIdFromClientY(e.clientY, initialTrackId);
        const targetLane = getTrackLaneById(targetTrackId);
        if (!targetLane) return;

        const laneRect = targetLane.getBoundingClientRect();
        const rawStart = (e.clientX - laneRect.left - pointerOffsetX) / pixelsPerSecond;
        const nextDuration = initialDuration;
        const snappedStart = normalizePlacementTime(rawStart, nextDuration);

        if (clipBox.parentElement !== targetLane) {
            targetLane.appendChild(clipBox);
        }

        clipBox.style.left = `${snappedStart * pixelsPerSecond}px`;
        clipBox.style.width = `${nextDuration * pixelsPerSecond}px`;
        placement.trackId = targetTrackId;
        placement.startTime = snappedStart;
        placement.duration = nextDuration;
        redrawPlacedClipWaveform(placement);
        return;
    }

    if (mode === 'resize') {
        const targetLane = getTrackLaneById(initialTrackId);
        if (!targetLane) return;

        const laneRect = targetLane.getBoundingClientRect();
        const laneX = (e.clientX - laneRect.left) / pixelsPerSecond;

        let nextStartTime = initialStartTime;
        let nextDuration = initialDuration;

        if (resizeSide === 'right') {
            const rawEnd = clamp(laneX, initialStartTime + minClipDuration, totalDuration);
            const snappedEnd = normalizePlacementTime(rawEnd, Math.max(minClipDuration, rawEnd - initialStartTime));
            nextDuration = clamp(snappedEnd - initialStartTime, minClipDuration, totalDuration - initialStartTime);
        } else if (resizeSide === 'left') {
            const rawStart = clamp(laneX, 0, initialStartTime + initialDuration - minClipDuration);
            const snappedStart = normalizePlacementTime(rawStart, Math.max(minClipDuration, initialStartTime + initialDuration - rawStart));
            nextStartTime = clamp(snappedStart, 0, initialStartTime + initialDuration - minClipDuration);
            nextDuration = clamp((initialStartTime + initialDuration) - nextStartTime, minClipDuration, totalDuration - nextStartTime);
        }

        placement.trackId = initialTrackId;
        placement.startTime = nextStartTime;
        placement.duration = nextDuration;

        clipBox.style.left = `${nextStartTime * pixelsPerSecond}px`;
        clipBox.style.width = `${nextDuration * pixelsPerSecond}px`;
        redrawPlacedClipWaveform(placement);
    }
}

function finishPlacedClipGesture(e) {
    if (!activeClipGesture || e.pointerId !== activeClipGesture.pointerId) return;

    const { placementId, clipBox, clipId } = activeClipGesture;
    const placement = getPlacementById(placementId);
    const clip = libraryClips.find(c => c.id === clipId);

    if (clipBox) {
        clipBox.classList.remove('dragging', 'resizing');
    }

    activeClipGesture = null;

    if (placement && clipBox && clip) {
        updatePlacedClipElement(placement, clipBox);
        redrawPlacedClipWaveform(placement);
    }

    if (isPlaying) {
        restartPlaybackAtCurrentTime();
    }

    updateExportButtonState();
}

function configureClipSource(source, clip, placement, offsetIntoClip = 0) {
    const bufferDuration = clip.audioBuffer.duration;
    const remainingDuration = placement.duration - offsetIntoClip;
    if (remainingDuration <= 0) return null;

    const shouldLoop = placement.duration > bufferDuration + 0.0001;
    const sourceOffset = shouldLoop ? (offsetIntoClip % bufferDuration) : clamp(offsetIntoClip, 0, bufferDuration);
    const playDuration = shouldLoop ? remainingDuration : Math.min(remainingDuration, bufferDuration - sourceOffset);

    source.buffer = clip.audioBuffer;
    source.loop = shouldLoop;
    if (shouldLoop) {
        source.loopStart = 0;
        source.loopEnd = bufferDuration;
    }

    if (playDuration <= 0) return null;

    return {
        sourceOffset,
        playDuration,
        shouldLoop,
    };
}

// Load AI Model in-browser (direct classes)
async function loadAIModel() {
    if (isAiLoading) return;
    isAiLoading = true;
    
    elEngineStatus.className = 'engine-status status-loading';
    elStatusText.textContent = 'Loading AI...';
    elProgressContainer.classList.remove('hidden');
    
    const progressItems = {};
    
    const progressCallback = (data) => {
        if (data.status === 'progress') {
            progressItems[data.file] = data.progress;
            
            // Calculate average progress
            let total = 0;
            let count = 0;
            for (const file in progressItems) {
                total += progressItems[file];
                count++;
            }
            const avg = count > 0 ? (total / count) : 0;
            const percent = Math.round(avg);
            
            elProgressPercent.textContent = `${percent}%`;
            elProgressBar.style.width = `${percent}%`;
            elProgressStatus.textContent = `Downloading AI files (${count} files loaded)...`;
        }
    };

    const model_id = 'Xenova/musicgen-small';
    try {
        console.log("Attempting to load AutoTokenizer and MusicgenForConditionalGeneration...");
        tokenizer = await AutoTokenizer.from_pretrained(model_id);
        
        // Prefer WebGPU on the current Transformers.js runtime. Older versions could
        // return valid-looking but noisy/static MusicGen tensors on some GPU stacks.
        model = await MusicgenForConditionalGeneration.from_pretrained(model_id, {
            device: 'webgpu',
            progress_callback: progressCallback,
            dtype: 'fp32',
        });
        
        elEngineStatus.className = 'engine-status status-ready';
        elStatusText.textContent = 'AI Ready (GPU)';
        console.log("WebGPU Model loaded successfully.");
    } catch (gpuError) {
        console.warn("WebGPU load failed. Falling back to WebAssembly (CPU):", gpuError);
        elProgressStatus.textContent = 'Falling back to CPU engine (WASM)...';
        
        try {
            model = await MusicgenForConditionalGeneration.from_pretrained(model_id, {
                device: 'wasm',
                progress_callback: progressCallback,
                dtype: 'fp32',
            });
            elEngineStatus.className = 'engine-status status-ready';
            elStatusText.textContent = 'AI Ready (CPU)';
            console.log("WASM CPU Model loaded successfully.");
        } catch (cpuError) {
            console.error("AI Model failed to load entirely:", cpuError);
            elEngineStatus.className = 'engine-status status-error';
            elStatusText.textContent = 'AI Load Failed';
            alert("Failed to load AI model. Please verify your internet connection and reload the page.");
        }
    } finally {
        isAiLoading = false;
        elProgressContainer.classList.add('hidden');
        if (model) {
            elBtnGenerate.disabled = false;
            elBtnGenerateText.textContent = 'Generate Loop';
        }
        updateExportButtonState();
    }
}

// Generate Sound Bite
async function generateSoundBite() {
    if (isGenerating || !model || !tokenizer) return;
    
    // Retrieve structured inputs
    const instrument = elInputInstrument.value.trim();
    const style = elInputStyle.value.trim();
    const mood = elInputMood.value.trim();
    const bpm = elInputBpm.value.trim();
    
    if (!instrument || !style || !mood || !bpm) {
        alert("Please fill in all four properties (Instrument, Style, Mood, BPM) to generate a loop!");
        return;
    }
    
    // Assemble structured prompt
    // Bias the model toward a single isolated instrument stem.
    const prompt = `${instrument} solo only, no other instruments, no accompaniment, no vocals, ${style}, ${mood} mood, ${bpm} bpm, seamless perfect loop, high quality`;
    const generationTake = createGenerationTake();
    const generationPrompt = `${prompt}, take ${generationTake}`;
    
    isGenerating = true;
    elBtnGenerate.disabled = true;
    elBtnGenerateText.textContent = 'Generating...';
    elEngineStatus.className = 'engine-status status-loading';
    elStatusText.textContent = 'Generating Audio...';
    
    elBtnGenerate.style.animation = 'pulse 1s infinite alternate';

    try {
        console.log(`Generating music for prompt: "${generationPrompt}"`);
        const ctx = ensureAudioContext();
        
        if (ctx.state === 'suspended') {
            await ctx.resume();
        }

        // Tokenize prompt
        const inputs = tokenizer(generationPrompt);

        // Run inference
        // Musicgen small runs at 50 Hz frame rate.
        // We want a perfect loop of exactly 5.0 seconds.
        // To do crossfading, we generate 5.4 seconds (270 tokens) and blend the extra 0.4s.
        const audio_values = await model.generate({
            ...inputs,
            max_new_tokens: 270,
            do_sample: true,
            guidance_scale: 4.5,
            temperature: 0.95,
            top_k: 120,
            top_p: 0.9,
        });

        const samplingRate = model.config.audio_encoder.sampling_rate || 32000;
        
        const audioData = extractMonoPcm(audio_values);
        
        console.log(`Audio extracted: ${audioData.length} samples @ ${samplingRate}Hz = ${(audioData.length/samplingRate).toFixed(2)}s | take: ${generationTake}`);

        // Convert float32 array to intermediate AudioBuffer
        const rawBuffer = ctx.createBuffer(1, audioData.length, samplingRate);
        rawBuffer.copyToChannel(audioData, 0);

        // Normalize raw buffer values
        normalizeAudioBuffer(rawBuffer);

        // Apply a 0.4s linear crossfade to create a perfect loop of exactly 5.0 seconds
        const processedBuffer = makeAudioBufferLoopable(rawBuffer, 5.0);

        // Create clip metadata object
        const clipId = `clip_${Date.now()}`;
        const newClip = {
            id: clipId,
            prompt: `${instrument} (${style}, ${mood})`,
            audioBuffer: processedBuffer,
            duration: clipDuration
        };

        // Store and render
        libraryClips.push(newClip);
        addClipToLibraryUI(newClip);
        updateLibraryUIState();
        
        // Auto scroll to the new clip in the library
        const elNewClipCard = document.getElementById(clipId);
        if (elNewClipCard) {
            elNewClipCard.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (err) {
        console.error("Audio generation failed:", err);
        alert(`Generation failed: ${err.message || err}`);
    } finally {
        isGenerating = false;
        elBtnGenerate.disabled = false;
        elBtnGenerateText.textContent = 'Generate Loop';
        elBtnGenerate.style.animation = '';
        
        if (model) {
            elEngineStatus.className = 'engine-status status-ready';
            elStatusText.textContent = 'AI Ready';
        }
    }
}

function createGenerationTake() {
    const seed = new Uint32Array(1);
    crypto.getRandomValues(seed);
    return seed[0].toString(36);
}

function extractMonoPcm(audioValues) {
    const tensor = audioValues?.audio_values || audioValues;

    if (!tensor || !tensor.data) {
        throw new Error('model.generate() returned no audio data.');
    }

    const dims = Array.isArray(tensor.dims) ? tensor.dims : [tensor.data.length];
    const data = tensor.data;
    const numSamples = dims[dims.length - 1];
    const channelCount = dims.length >= 2 ? dims[dims.length - 2] : 1;
    const expectedLength = dims.reduce((total, dim) => total * dim, 1);

    console.log('audio_values.dims:', dims, '| data.length:', data.length, '| expected:', expectedLength);

    if (!Number.isFinite(numSamples) || numSamples <= 0 || data.length < numSamples) {
        throw new Error(`Invalid generated audio shape: [${dims.join(', ')}].`);
    }

    // Common MusicGen output is [batch, channels, samples]. Use the first batch,
    // and intentionally downmix stereo/multi-channel output instead of flattening it.
    const batchStride = channelCount * numSamples;
    const batchOffset = data.length >= batchStride ? 0 : Math.max(0, data.length - numSamples);
    const mono = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        let mixedSample = 0;
        for (let channel = 0; channel < channelCount; channel++) {
            const sample = Number(data[batchOffset + channel * numSamples + i]);
            mixedSample += Number.isFinite(sample) ? sample : 0;
        }
        mono[i] = mixedSample / channelCount;
    }

    const stats = getAudioStats(mono);
    console.log('Audio stats:', stats);

    if (stats.nonFinite > 0) {
        throw new Error('Generated audio contains invalid sample values.');
    }

    if (stats.peak < 0.0001) {
        throw new Error('Generated audio is silent. Try reloading the model and generating again.');
    }

    return mono;
}

function getAudioStats(samples) {
    let peak = 0;
    let rmsTotal = 0;
    let nonFinite = 0;

    for (let i = 0; i < samples.length; i++) {
        const sample = samples[i];
        if (!Number.isFinite(sample)) {
            nonFinite++;
            continue;
        }

        const absSample = Math.abs(sample);
        peak = Math.max(peak, absSample);
        rmsTotal += sample * sample;
    }

    return {
        peak,
        rms: Math.sqrt(rmsTotal / samples.length),
        nonFinite,
    };
}

// Loop processor: Crossfades the tail of rawBuffer back into the head
// Outputs a buffer that is exactly targetDuration long
function makeAudioBufferLoopable(rawBuffer, targetDuration = 5.0) {
    const sampleRate = rawBuffer.sampleRate;
    const numChannels = rawBuffer.numberOfChannels;
    const totalSamples = rawBuffer.length;
    
    const targetSamples = Math.floor(targetDuration * sampleRate);
    const fadeSamples = totalSamples - targetSamples; // Extra samples generated for blending
    
    if (fadeSamples <= 0 || totalSamples <= fadeSamples * 2) {
        console.warn("Audio buffer too short or exact target size; skipped crossfade.");
        return rawBuffer;
    }
    
    console.log(`Crossfading loop: blending last ${fadeSamples} samples (${(fadeSamples/sampleRate).toFixed(3)}s) back into start.`);

    // Create a new buffer matching the exact target duration
    const loopedBuffer = audioCtx.createBuffer(numChannels, targetSamples, sampleRate);
    
    for (let channel = 0; channel < numChannels; channel++) {
        const oldData = rawBuffer.getChannelData(channel);
        const newData = loopedBuffer.getChannelData(channel);
        
        // 1. Copy the steady state section (from fadeSamples onwards)
        for (let i = fadeSamples; i < targetSamples; i++) {
            newData[i] = oldData[i];
        }
        
        // 2. Linear crossfade blend the tail (indices targetSamples to totalSamples - 1)
        //    into the head (indices 0 to fadeSamples - 1)
        for (let i = 0; i < fadeSamples; i++) {
            const alpha = i / fadeSamples; // Ramp from 0.0 to 1.0
            const tailSample = oldData[targetSamples + i];
            const headSample = oldData[i];
            
            // Linear mix
            newData[i] = (1.0 - alpha) * tailSample + alpha * headSample;
        }
    }
    
    return loopedBuffer;
}

// Normalize Audio Buffer values to Peak Amplitude of 0.95
function normalizeAudioBuffer(buffer) {
    const channelData = buffer.getChannelData(0);
    let maxVal = 0;
    
    for (let i = 0; i < channelData.length; i++) {
        const absVal = Math.abs(channelData[i]);
        if (absVal > maxVal) {
            maxVal = absVal;
        }
    }
    
    if (maxVal > 0) {
        const scaleFactor = 0.95 / maxVal;
        for (let i = 0; i < channelData.length; i++) {
            channelData[i] *= scaleFactor;
        }
    }
}

// UI Setup
function setupUIEventListeners() {
    // Generate loop button
    elBtnGenerate.addEventListener('click', generateSoundBite);

    // Dynamic suggestions chips setup
    document.querySelectorAll('.input-chips span').forEach(chip => {
        chip.addEventListener('click', () => {
            const targetId = chip.parentElement.getAttribute('data-target');
            const targetInput = document.getElementById(targetId);
            if (targetInput) {
                targetInput.value = chip.textContent;
                
                // Chip tap flash animation
                chip.style.background = 'var(--clr-drums)';
                chip.style.color = 'var(--bg-base)';
                setTimeout(() => {
                    chip.style.background = '';
                    chip.style.color = '';
                }, 300);
            }
        });
    });

    // Project BPM slider listener
    elBpmInput.addEventListener('input', () => {
        updateTempoSettings();
        if (isPlaying) {
            restartPlaybackAtCurrentTime();
        }
    });

    // Loop BPM Sync button click
    elSyncBpmBtn.addEventListener('click', () => {
        elInputBpm.value = elBpmInput.value;
        // Visual indicator pulse
        elSyncBpmBtn.style.background = 'var(--clr-drums)';
        elSyncBpmBtn.style.color = 'var(--bg-base)';
        setTimeout(() => {
            elSyncBpmBtn.style.background = '';
            elSyncBpmBtn.style.color = '';
        }, 300);
    });

    // Playback Controls
    elBtnPlay.addEventListener('click', () => startPlayback());
    elBtnPause.addEventListener('click', () => pausePlayback());
    elBtnStop.addEventListener('click', () => stopPlayback());
    
    // Loop Button
    elBtnLoop.addEventListener('click', () => {
        isLooping = !isLooping;
        elBtnLoop.classList.toggle('active', isLooping);
    });

    window.addEventListener('pointermove', updatePlacedClipGesture);
    window.addEventListener('pointerup', finishPlacedClipGesture);
    window.addEventListener('pointercancel', finishPlacedClipGesture);

    // Export button
    elBtnExport.addEventListener('click', exportComposition);

    // Help modal controls
    elBtnHelp.addEventListener('click', () => elHelpModal.classList.remove('hidden'));
    elBtnCloseHelp.addEventListener('click', () => elHelpModal.classList.add('hidden'));
    elBtnGetStarted.addEventListener('click', () => elHelpModal.classList.add('hidden'));

    // Track controls setup (mute, solo, volume)
    const trackRows = document.querySelectorAll('.daw-track-row');
    trackRows.forEach((row, index) => {
        const btnMute = row.querySelector('.mute-btn');
        const btnSolo = row.querySelector('.solo-btn');
        const inputVol = row.querySelector('.track-volume');

        btnMute.addEventListener('click', () => toggleTrackMute(index, btnMute));
        btnSolo.addEventListener('click', () => toggleTrackSolo(index, btnSolo));
        inputVol.addEventListener('input', (e) => adjustTrackVolume(index, parseFloat(e.target.value)));
    });

    // Snap grid logic mapping
    elTrackLanes.forEach(lane => {
        lane.addEventListener('dragover', (e) => {
            e.preventDefault();
            lane.classList.add('drag-over');
        });

        lane.addEventListener('dragleave', () => {
            lane.classList.remove('drag-over');
        });

        lane.addEventListener('drop', (e) => {
            lane.classList.remove('drag-over');
            e.preventDefault();
            
            const clipId = e.dataTransfer.getData('text/plain');
            const dragSourcePlacedId = e.dataTransfer.getData('placed-id');
            const trackId = parseInt(lane.getAttribute('data-track-id'));
            
            const rect = lane.getBoundingClientRect();
            const dropX = e.clientX - rect.left;
            let timeInSeconds = dropX / pixelsPerSecond;
            
            // Grid Snapping logic based on settings (BPM)
            const snapMode = elSnapSelect.value;
            if (snapMode === 'beat') {
                timeInSeconds = Math.round(timeInSeconds / secondsPerBeat) * secondsPerBeat;
            } else if (snapMode === 'bar') {
                timeInSeconds = Math.round(timeInSeconds / secondsPerBar) * secondsPerBar;
            }
            timeInSeconds = clamp(timeInSeconds, 0, totalDuration);
            
            if (dragSourcePlacedId) {
                movePlacedClip(dragSourcePlacedId, trackId, normalizePlacementTime(timeInSeconds, clipDuration));
            } else {
                // DROP from library — create a new placement
                const clip = libraryClips.find(c => c.id === clipId);
                if (clip) {
                    placeClipOnTimeline(clip, trackId, normalizePlacementTime(timeInSeconds, clipDuration));
                }
            }
        });
    });

    // Seek timeline click listener (ruler)
    elRuler.addEventListener('click', (e) => {
        const elDawGrid = document.getElementById('daw-grid');
        const rect = elRuler.getBoundingClientRect();
        // Add scrollLeft because ruler is visually shifted but click coords are unshifted
        const clickX = e.clientX - rect.left + elDawGrid.scrollLeft;
        let newTime = clickX / pixelsPerSecond;
        if (newTime < 0) newTime = 0;
        if (newTime > totalDuration) newTime = totalDuration;
        seekPlayback(newTime);
    });

    // Sync ruler horizontal scroll with daw-grid scroll
    const elDawGrid = document.getElementById('daw-grid');
    elDawGrid.addEventListener('scroll', () => {
        elRuler.style.transform = `translateX(-${elDawGrid.scrollLeft}px)`;
    });

    // Keyboard controls
    window.addEventListener('keydown', (e) => {
        if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') {
            return;
        }

        if (e.code === 'Space') {
            e.preventDefault();
            if (isPlaying) {
                pausePlayback();
            } else {
                startPlayback();
            }
        } else if (e.code === 'Delete' || e.code === 'Backspace') {
            const selectedElement = document.querySelector('.placed-clip.selected');
            if (selectedElement) {
                const placedId = selectedElement.getAttribute('data-placed-id');
                removePlacedClip(placedId);
            }
        }
    });
}

// Track control nodes
function toggleTrackMute(trackId, btnElement) {
    trackMuteStates[trackId] = !trackMuteStates[trackId];
    btnElement.classList.toggle('active', trackMuteStates[trackId]);
    updateTrackRoutingGains();
}

function toggleTrackSolo(trackId, btnElement) {
    trackSoloStates[trackId] = !trackSoloStates[trackId];
    btnElement.classList.toggle('active', trackSoloStates[trackId]);
    updateTrackRoutingGains();
}

function adjustTrackVolume(trackId, value) {
    if (trackVolumeNodes[trackId]) {
        trackVolumeNodes[trackId]._sliderValue = value;
        updateTrackRoutingGains();
    }
}

function updateTrackRoutingGains() {
    const isAnySoloActive = trackSoloStates.some(state => state === true);
    
    for (let i = 0; i < numTracks; i++) {
        const gainNode = trackVolumeNodes[i];
        if (!gainNode) continue;
        
        const sliderValue = gainNode._sliderValue !== undefined ? gainNode._sliderValue : 0.8;
        let targetGain = sliderValue;
        
        if (trackMuteStates[i]) {
            targetGain = 0;
        } else if (isAnySoloActive && !trackSoloStates[i]) {
            targetGain = 0;
        }
        
        if (audioCtx) {
            gainNode.gain.setTargetAtTime(targetGain, audioCtx.currentTime, 0.015);
        } else {
            gainNode.gain.value = targetGain;
        }
    }
}

// Draw timeline ruler — major marks every clipDuration (5s) to align with placed clips,
// minor beat marks in between so users can still feel the groove.
function renderTimelineRuler() {
    elRuler.innerHTML = '';
    const laneWidth = totalDuration * pixelsPerSecond;
    elRuler.style.width = `${laneWidth}px`;
    
    elTrackLanes.forEach(lane => {
        lane.style.width = `${laneWidth}px`;
    });

    // Major sections every 5 seconds (= 1 clip width)
    const sectionDuration = clipDuration; // 5 seconds
    const numSections = Math.ceil(totalDuration / sectionDuration);

    // Also draw minor beat marks between sections
    const totalBeats = Math.ceil(totalDuration / secondsPerBeat);
    
    for (let b = 0; b <= totalBeats; b++) {
        const timeInSeconds = b * secondsPerBeat;
        if (timeInSeconds > totalDuration) break;
        
        const isSection = Math.abs((timeInSeconds % sectionDuration)) < 0.001 || 
                          Math.abs((timeInSeconds % sectionDuration) - sectionDuration) < 0.001;
        
        if (isSection) continue; // drawn separately below for correct z-ordering
        
        const tick = document.createElement('div');
        tick.className = 'ruler-tick minor';
        tick.style.left = `${timeInSeconds * pixelsPerSecond}px`;
        elRuler.appendChild(tick);
    }

    // Draw major section marks on top (every 5s = 1 clip)
    for (let s = 0; s <= numSections; s++) {
        const timeInSeconds = s * sectionDuration;
        if (timeInSeconds > totalDuration + 0.001) break;
        
        const tick = document.createElement('div');
        tick.className = 'ruler-tick major';
        tick.style.left = `${timeInSeconds * pixelsPerSecond}px`;
        tick.innerHTML = `<span>${s + 1}</span>`;
        elRuler.appendChild(tick);
    }
}

// Library list controller
function updateLibraryUIState() {
    const count = libraryClips.length;
    elLibraryCount.textContent = count;
    
    if (count > 0) {
        elLibraryEmpty.style.display = 'none';
        elLibraryList.style.display = 'flex';
    } else {
        elLibraryEmpty.style.display = 'flex';
        elLibraryList.style.display = 'none';
    }
}

// Add a generated clip to Library sidebar
function addClipToLibraryUI(clip) {
    const card = document.createElement('div');
    card.className = 'library-clip-item';
    card.id = clip.id;
    card.draggable = true;
    
    card.innerHTML = `
        <div class="clip-top">
            <div class="clip-title-container">
                <div class="clip-prompt" title="${clip.prompt}">${clip.prompt}</div>
                <div class="clip-meta">5.0s Loop | 32kHz</div>
            </div>
            <div class="clip-actions">
                <button class="clip-action-btn play-clip-btn" title="Play Clip Preview">
                    <span class="material-icons">play_arrow</span>
                </button>
                <button class="clip-action-btn delete-btn" title="Delete Clip">
                    <span class="material-icons">delete</span>
                </button>
            </div>
        </div>
        <canvas class="clip-waveform-canvas"></canvas>
    `;
    
    const canvas = card.querySelector('.clip-waveform-canvas');
    setTimeout(() => drawWaveform(clip.audioBuffer, canvas, 'var(--clr-drums)'), 50);

    card.addEventListener('dragstart', (e) => {
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'copyMove';
        e.dataTransfer.setData('text/plain', clip.id);
    });

    card.addEventListener('dragend', () => {
        card.classList.remove('dragging');
    });

    let previewSource = null;
    let isPreviewPlaying = false;
    const btnPlay = card.querySelector('.play-clip-btn');
    
    const stopPreview = () => {
        if (previewSource) {
            try { previewSource.stop(); } catch(e){}
            previewSource = null;
        }
        isPreviewPlaying = false;
        btnPlay.innerHTML = '<span class="material-icons">play_arrow</span>';
        btnPlay.title = 'Play Clip Preview';
    };

    btnPlay.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        if (isPlaying) stopPlayback();

        if (isPreviewPlaying) {
            stopPreview();
        } else {
            document.querySelectorAll('.play-clip-btn').forEach(btn => {
                if (btn !== btnPlay && btn.innerHTML.includes('stop')) {
                    btn.click();
                }
            });

            if (audioCtx.state === 'suspended') {
                await audioCtx.resume();
            }

            previewSource = audioCtx.createBufferSource();
            previewSource.buffer = clip.audioBuffer;
            previewSource.connect(masterGainNode);
            previewSource.onended = () => stopPreview();
            
            previewSource.start(0);
            isPreviewPlaying = true;
            btnPlay.innerHTML = '<span class="material-icons">stop</span>';
            btnPlay.title = 'Stop Preview';
        }
    });

    card.querySelector('.delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        stopPreview();
        
        if (confirm("Delete this loop? It will be removed from library and tracks.")) {
            const index = libraryClips.indexOf(clip);
            if (index > -1) libraryClips.splice(index, 1);
            
            const placementsToRemove = placedClips.filter(pc => pc.clipId === clip.id);
            placementsToRemove.forEach(pc => removePlacedClip(pc.id, false));
            
            card.remove();
            updateLibraryUIState();
            updateExportButtonState();
        }
    });

    elLibraryList.appendChild(card);
}

// Waveform drawing
function drawWaveform(audioBuffer, canvas, color, options = {}) {
    const repeatDuration = options.repeatDuration;
    const ctx = canvas.getContext('2d');
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);
    
    const channelData = audioBuffer.getChannelData(0);
    const amp = height / 2;
    
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const shouldTile = Number.isFinite(repeatDuration) && repeatDuration > audioBuffer.duration + 0.0001;

    if (shouldTile) {
        const cycleWidth = Math.max(1, width * (audioBuffer.duration / repeatDuration));
        const samplesPerCyclePixel = channelData.length / cycleWidth;

        for (let i = 0; i < width; i++) {
            const localX = i % cycleWidth;
            const startSample = Math.floor(localX * samplesPerCyclePixel);
            const endSample = Math.max(startSample + 1, Math.min(channelData.length, Math.ceil((localX + 1) * samplesPerCyclePixel)));

            let min = 1.0;
            let max = -1.0;

            for (let j = startSample; j < endSample; j++) {
                const datum = channelData[j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }

            const yTop = (1 + min) * amp;
            const yBottom = (1 + max) * amp;

            ctx.moveTo(i, Math.max(1, yTop));
            ctx.lineTo(i, Math.min(height - 1, yBottom));
        }
    } else {
        const step = Math.max(1, Math.ceil(channelData.length / width));

        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;

            for (let j = 0; j < step; j++) {
                const datum = channelData[(i * step) + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }

            const yTop = (1 + min) * amp;
            const yBottom = (1 + max) * amp;

            ctx.moveTo(i, Math.max(1, yTop));
            ctx.lineTo(i, Math.min(height - 1, yBottom));
        }
    }
    
    ctx.stroke();
}

// Place clip on grid
function placeClipOnTimeline(clip, trackId, startTime) {
    const resolvedStartTime = resolvePlacementStart(trackId, startTime, clipDuration);
    const placedId = `placed_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    
    const newPlacement = {
        id: placedId,
        clipId: clip.id,
        trackId: trackId,
        startTime: resolvedStartTime,
        duration: clipDuration
    };

    placedClips.push(newPlacement);
    renderPlacedClipUI(newPlacement, clip);
    updateExportButtonState();
    
    if (isPlaying) {
        restartPlaybackAtCurrentTime();
    }
}

function resolvePlacementStart(trackId, desiredStartTime, duration = clipDuration, excludePlacementId = null) {
    const maxStartTime = Math.max(0, totalDuration - duration);
    return clamp(snapTimeToGrid(clamp(desiredStartTime, 0, maxStartTime)), 0, maxStartTime);
}

// Render placed clip inside DAW lane
function renderPlacedClipUI(placement, clip) {
    const lane = document.querySelector(`.track-lane[data-track-id="${placement.trackId}"]`);
    if (!lane) return;

    const clipBox = document.createElement('div');
    clipBox.className = 'placed-clip';
    clipBox.id = placement.id;
    clipBox.setAttribute('data-placed-id', placement.id);
    updatePlacedClipElement(placement, clipBox);
    
    clipBox.innerHTML = `
        <div class="clip-resize-handle clip-resize-handle-left" title="Trim start"></div>
        <div class="placed-clip-header">
            <span class="placed-clip-name" title="${clip.prompt}">${clip.prompt}</span>
            <button class="btn-remove-clip" title="Delete Loop">
                <span class="material-icons" style="font-size: 13px;">close</span>
            </button>
        </div>
        <canvas class="placed-clip-waveform"></canvas>
        <div class="clip-resize-handle clip-resize-handle-right" title="Loop or trim end"></div>
    `;

    setTimeout(() => redrawPlacedClipWaveform(placement), 50);

    clipBox.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.placed-clip').forEach(box => {
            box.classList.remove('selected');
        });
        clipBox.classList.add('selected');
    });

    lane.addEventListener('click', () => {
        clipBox.classList.remove('selected');
    });

    clipBox.querySelector('.btn-remove-clip').addEventListener('click', (e) => {
        e.stopPropagation();
        removePlacedClip(placement.id);
    });

    clipBox.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        if (e.target.closest('.btn-remove-clip')) return;
        const resizeHandle = e.target.closest('.clip-resize-handle');
        const mode = resizeHandle ? 'resize' : 'move';
        const resizeSide = resizeHandle?.classList.contains('clip-resize-handle-left') ? 'left' : 'right';
        beginPlacedClipGesture(e, placement.id, mode, resizeHandle ? resizeSide : null);
    });

    lane.appendChild(clipBox);
}

// Remove clip from DAW timeline
function removePlacedClip(placedId, redraw = true) {
    const idx = placedClips.findIndex(pc => pc.id === placedId);
    if (idx > -1) {
        placedClips.splice(idx, 1);
        
        const el = document.getElementById(placedId);
        if (el) el.remove();
        
        updateExportButtonState();
        
        if (isPlaying && redraw) {
            restartPlaybackAtCurrentTime();
        }
    }
}

// Move an existing placed clip to a new track/time (drag-to-reposition)
function movePlacedClip(placedId, newTrackId, newStartTime) {
    const placement = placedClips.find(pc => pc.id === placedId);
    if (!placement) return;
    
    const clip = libraryClips.find(c => c.id === placement.clipId);
    if (!clip) return;

    placement.trackId = newTrackId;
    placement.startTime = resolvePlacementStart(newTrackId, newStartTime, placement.duration, placedId);

    const clipBox = document.getElementById(placedId);
    if (clipBox) {
        const targetLane = getTrackLaneById(newTrackId);
        if (targetLane && clipBox.parentElement !== targetLane) {
            targetLane.appendChild(clipBox);
        }
        updatePlacedClipElement(placement, clipBox);
        redrawPlacedClipWaveform(placement);
    } else {
        renderPlacedClipUI(placement, clip);
    }
    
    if (isPlaying) {
        restartPlaybackAtCurrentTime();
    }
}

// Global Playback Engine
async function startPlayback() {
    if (isPlaying) return;
    
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }
    
    isPlaying = true;
    elBtnPlay.classList.add('active');
    playStartTime = audioCtx.currentTime;
    
    scheduleClips(pauseOffset);
    runPlayheadAnimation();
}

function pausePlayback() {
    if (!isPlaying) return;
    
    isPlaying = false;
    elBtnPlay.classList.remove('active');
    pauseOffset += (audioCtx.currentTime - playStartTime);
    
    cancelAnimationFrame(animationFrameId);
    stopActiveSources();
}

function stopPlayback() {
    isPlaying = false;
    elBtnPlay.classList.remove('active');
    
    pauseOffset = 0;
    playheadPosition = 0;
    elPlayhead.style.left = `${headerWidth}px`;
    
    cancelAnimationFrame(animationFrameId);
    stopActiveSources();
}

function seekPlayback(newTime) {
    const wasPlaying = isPlaying;
    
    if (isPlaying) {
        stopActiveSources();
        isPlaying = false;
    }
    
    pauseOffset = newTime;
    playheadPosition = newTime;
    elPlayhead.style.left = `${headerWidth + newTime * pixelsPerSecond}px`;
    
    if (wasPlaying) {
        startPlayback();
    }
}

function restartPlaybackAtCurrentTime() {
    const currentPos = playheadPosition;
    stopActiveSources();
    playStartTime = audioCtx.currentTime;
    pauseOffset = currentPos;
    scheduleClips(currentPos);
}

function stopActiveSources() {
    activeAudioSources.forEach(src => {
        try {
            src.stop();
        } catch(e) {}
    });
    activeAudioSources = [];
}

function scheduleClips(offsetTime) {
    stopActiveSources();
    const now = audioCtx.currentTime;
    
    placedClips.forEach(placement => {
        const clip = libraryClips.find(c => c.id === placement.clipId);
        if (!clip) return;
        
        const clipStart = placement.startTime;
        const clipEnd = clipStart + placement.duration;
        const source = audioCtx.createBufferSource();
        source.connect(trackVolumeNodes[placement.trackId]);

        if (clipStart >= offsetTime) {
            const delay = clipStart - offsetTime;
            const scheduled = configureClipSource(source, clip, placement, 0);
            if (scheduled) {
                source.start(now + delay, scheduled.sourceOffset, scheduled.playDuration);
                activeAudioSources.push(source);
            }
        } else if (clipStart < offsetTime && clipEnd > offsetTime) {
            const playOffset = offsetTime - clipStart;
            const scheduled = configureClipSource(source, clip, placement, playOffset);
            if (scheduled) {
                source.start(now, scheduled.sourceOffset, scheduled.playDuration);
                activeAudioSources.push(source);
            }
        }
    });
}

function runPlayheadAnimation() {
    if (!isPlaying) return;
    
    const elapsed = audioCtx.currentTime - playStartTime;
    playheadPosition = pauseOffset + elapsed;
    
    if (playheadPosition >= totalDuration) {
        if (isLooping) {
            stopActiveSources();
            playStartTime = audioCtx.currentTime;
            pauseOffset = 0;
            playheadPosition = 0;
            scheduleClips(0);
        } else {
            stopPlayback();
            return;
        }
    }
    
    elPlayhead.style.left = `${headerWidth + playheadPosition * pixelsPerSecond}px`;
    animationFrameId = requestAnimationFrame(runPlayheadAnimation);
}

function updateExportButtonState() {
    const hasClips = placedClips.length > 0;
    elBtnExport.disabled = !hasClips;
}

// Stereo render mixdown WAV download
async function exportComposition() {
    if (placedClips.length === 0) return;

    elBtnExport.disabled = true;
    const originalText = elBtnExport.innerHTML;
    elBtnExport.innerHTML = '<span class="material-icons">hourglass_empty</span> Rendering...';

    try {
        const sampleRate = 44100;
        const offlineCtx = new OfflineAudioContext(2, sampleRate * totalDuration, sampleRate);

        const offlineTrackGains = [];
        const offlineMasterGain = offlineCtx.createGain();
        offlineMasterGain.gain.value = 0.95;
        offlineMasterGain.connect(offlineCtx.destination);

        for (let i = 0; i < numTracks; i++) {
            const gainNode = offlineCtx.createGain();
            const sliderVal = trackVolumeNodes[i]._sliderValue !== undefined ? trackVolumeNodes[i]._sliderValue : 0.8;
            
            const isAnySoloActive = trackSoloStates.some(state => state === true);
            let gainVal = sliderVal;
            
            if (trackMuteStates[i]) {
                gainVal = 0;
            } else if (isAnySoloActive && !trackSoloStates[i]) {
                gainVal = 0;
            }

            gainNode.gain.value = gainVal;
            gainNode.connect(offlineMasterGain);
            offlineTrackGains.push(gainNode);
        }

        placedClips.forEach(placement => {
            const clip = libraryClips.find(c => c.id === placement.clipId);
            if (!clip) return;

            const source = offlineCtx.createBufferSource();
            source.connect(offlineTrackGains[placement.trackId]);
            const scheduled = configureClipSource(source, clip, placement, 0);
            if (scheduled) {
                source.start(placement.startTime, scheduled.sourceOffset, scheduled.playDuration);
            }
        });

        const renderedBuffer = await offlineCtx.startRendering();
        const wavBlob = bufferToWav(renderedBuffer);

        const url = URL.createObjectURL(wavBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `Acoustix_Composition_${Date.now().toString().slice(-6)}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

    } catch (err) {
        console.error("Render failed:", err);
        alert("Export failed: " + err);
    } finally {
        elBtnExport.innerHTML = originalText;
        updateExportButtonState();
    }
}

// Convert AudioBuffer to WAV format (16-bit stereo PCM)
function bufferToWav(buffer) {
    let numOfChan = buffer.numberOfChannels,
        length = buffer.length * numOfChan * 2 + 44,
        bufferArr = new ArrayBuffer(length),
        view = new DataView(bufferArr),
        channels = [], i, sample,
        offset = 0,
        pos = 0;

    function setUint16(data) {
        view.setUint16(pos, data, true);
        pos += 2;
    }

    function setUint32(data) {
        view.setUint32(pos, data, true);
        pos += 4;
    }

    setUint32(0x46464952);                         // "RIFF"
    setUint32(length - 8);                         // file length - 8
    setUint32(0x45564157);                         // "WAVE"
    setUint32(0x20746d66);                         // "fmt " chunk
    setUint32(16);                                 // chunk length
    setUint16(1);                                  // sample format (raw PCM)
    setUint16(numOfChan);                          // channel count
    setUint32(buffer.sampleRate);                  // sample rate
    setUint32(buffer.sampleRate * 2 * numOfChan);  // byte rate
    setUint16(numOfChan * 2);                      // block align
    setUint16(16);                                 // bits per sample
    setUint32(0x61746164);                         // "data" chunk
    setUint32(length - pos - 4);                   // chunk length

    for (i = 0; i < buffer.numberOfChannels; i++) {
        channels.push(buffer.getChannelData(i));
    }

    while (pos < length - 4) {
        for (i = 0; i < numOfChan; i++) {             // interleave channels
            sample = Math.max(-1, Math.min(1, channels[i][offset])); 
            sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF); 
            view.setInt16(pos, sample, true); 
            pos += 2;
        }
        offset++;
    }

    return new Blob([bufferArr], { type: 'audio/wav' });
}
