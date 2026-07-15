function CMBAudio() {

    this.context = null;
    this.masterGain = null;

    this.noiseSource = null;
    this.blueNoiseSource = null;
    this.blueNoiseGain = null;

    this.dryGain = null;
    this.invertedGain = null;
    this.filters = [];

    this.hotness = 0.5;
    this.masterVolume = 0.5;

    this.eqSettings = [
        { frequency: 220,  gain: 12.0, q: 2.3 },
        { frequency: 537,  gain: 8.6,  q: 5.1 },
        { frequency: 810,  gain: 8.4,  q: 7.4 },
        { frequency: 1120, gain: 5.2,  q: 9.3 },
        { frequency: 1440, gain: 3.5,  q: 11.0 }
    ];

    this.createPinkNoiseBuffer = function() {

        let bufferSize = 2 * this.context.sampleRate;
        let buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
        let output = buffer.getChannelData(0);

        let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;

        for (let i = 0; i < bufferSize; i++) {
            let white = Math.random() * 2 - 1;

            b0 = 0.99886 * b0 + white * 0.0555179;
            b1 = 0.99332 * b1 + white * 0.0750759;
            b2 = 0.96900 * b2 + white * 0.1538520;
            b3 = 0.86650 * b3 + white * 0.3104856;
            b4 = 0.55000 * b4 + white * 0.5329522;
            b5 = -0.7616 * b5 - white * 0.0168980;

            output[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
            output[i] *= 0.08;

            b6 = white * 0.115926;
        }

        return buffer;
    };

    this.createBlueNoiseBuffer = function() {

        let bufferSize = 2 * this.context.sampleRate;
        let buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
        let output = buffer.getChannelData(0);

        let previousWhite = 0;

        for (let i = 0; i < bufferSize; i++) {
            let white = Math.random() * 2 - 1;

            // Blue-ish noise: emphasize rapid changes.
            output[i] = (white - previousWhite) * 0.08;

            previousWhite = white;
        }

        return buffer;
    };

    this.startTone = function() {

        if (this.context) return;

        this.context = new (window.AudioContext || window.webkitAudioContext)();

        this.masterGain = this.context.createGain();
        this.masterGain.gain.value = this.masterVolume;
        this.masterGain.connect(this.context.destination);

        this.noiseSource = this.context.createBufferSource();
        this.noiseSource.buffer = this.createPinkNoiseBuffer();
        this.noiseSource.loop = true;

        this.dryGain = this.context.createGain();
        this.dryGain.gain.value = 1.0;

        this.invertedGain = this.context.createGain();
        this.invertedGain.gain.value = -1.0;

        this.noiseSource.connect(this.dryGain);
        this.dryGain.connect(this.masterGain);

        let previous = this.noiseSource;

        this.filters = [];

        for (let i = 0; i < this.eqSettings.length; i++) {

            let setting = this.eqSettings[i];

            let filter = this.context.createBiquadFilter();
            filter.type = "peaking";
            filter.frequency.value = setting.frequency;
            filter.gain.value = setting.gain;
            filter.Q.value = setting.q;

            previous.connect(filter);
            previous = filter;

            this.filters.push(filter);
        }

        previous.connect(this.invertedGain);
        this.invertedGain.connect(this.masterGain);

        this.blueNoiseSource = this.context.createBufferSource();
        this.blueNoiseSource.buffer = this.createBlueNoiseBuffer();
        this.blueNoiseSource.loop = true;

        this.blueNoiseGain = this.context.createGain();
        this.blueNoiseGain.gain.value = 0.0;

        this.blueNoiseSource.connect(this.blueNoiseGain);
        this.blueNoiseGain.connect(this.masterGain);

        this.noiseSource.start();
        this.blueNoiseSource.start();

        this.setHotness(this.hotness);
    };

    this.stopTone = function() {

        if (this.noiseSource) {
            this.noiseSource.stop();
        }

        if (this.blueNoiseSource) {
            this.blueNoiseSource.stop();
        }

        this.noiseSource = null;
        this.blueNoiseSource = null;
        this.blueNoiseGain = null;

        this.dryGain = null;
        this.invertedGain = null;
        this.filters = [];

        this.masterGain = null;
        this.context = null;
    };

    this.setSpectrumEQ = function(settings) {

        this.eqSettings = settings;

        if (!this.context || this.filters.length === 0) return;

        var now = this.context.currentTime;
        var rampTime = 0.15;

        for (let i = 0; i < settings.length && i < this.filters.length; i++) {

            this.filters[i].frequency.cancelScheduledValues(now);
            this.filters[i].gain.cancelScheduledValues(now);
            this.filters[i].Q.cancelScheduledValues(now);

            this.filters[i].frequency.setTargetAtTime(settings[i].frequency, now, rampTime);
            this.filters[i].gain.setTargetAtTime(settings[i].gain, now, rampTime);
            this.filters[i].Q.setTargetAtTime(settings[i].q, now, rampTime);
        }
    };

    this.setHotness = function(hotness) {

        this.hotness = Math.max(0, Math.min(1, hotness));

        if (!this.context || !this.blueNoiseGain) return;

        var now = this.context.currentTime;
        var rampTime = 0.2;

        var blueLevel = 0.18 * Math.max(0, this.hotness - 0.5) * 2;
        blueLevel = Math.max(0, Math.min(0.18, blueLevel));

        this.blueNoiseGain.gain.cancelScheduledValues(now);
        this.blueNoiseGain.gain.setTargetAtTime(blueLevel, now, rampTime);
    };

    this.testHotness = function() {
        this.setHotness(1.0);
    };

    this.resetHotness = function() {
        this.setHotness(0.5);
    };

    this.testEQ = function() {

        this.setSpectrumEQ([
            { frequency: 220,  gain: 20, q: 2 },
            { frequency: 537,  gain: 0,  q: 5 },
            { frequency: 810,  gain: 0,  q: 5 },
            { frequency: 1120, gain: 0,  q: 5 },
            { frequency: 1440, gain: 0,  q: 5 }
        ]);
    };

    this.resetEQ = function() {

        this.setSpectrumEQ([
            { frequency: 220,  gain: 12.0, q: 2.3 },
            { frequency: 537,  gain: 8.6,  q: 5.1 },
            { frequency: 810,  gain: 8.4,  q: 7.4 },
            { frequency: 1120, gain: 5.2,  q: 9.3 },
            { frequency: 1440, gain: 3.5,  q: 11.0 }
        ]);
    };

    this.setVolume = function(value) {

        value = Math.max(0, Math.min(1, value));

        this.masterVolume = value;

        if (this.masterGain) {
            this.masterGain.gain.value = this.masterVolume;
        }
    };

    this.setAmplitudes = function(amplitudes) {};
}