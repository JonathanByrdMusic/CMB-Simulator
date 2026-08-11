function CMBAudio() {

    // ---------------------------------------------------------
    // Core audio state
    // ---------------------------------------------------------

    this.context = null;
    this.masterGain = null;

    this.temperatureBus = null;
    this.polarizationBus = null;

    this.masterVolume = 0.5;
    this.observationMode = "temperature";


    // ---------------------------------------------------------
    // Temperature sonification
    // ---------------------------------------------------------

    this.noiseSource = null;

    this.dryGain = null;
    this.invertedGain = null;
    this.filters = [];

    this.blueNoiseSource = null;
    this.blueNoiseGain = null;

    this.hotness = 0.5;


    // ---------------------------------------------------------
    // Polarization sonification
    // ---------------------------------------------------------

    this.polarizationCenterSource = null;
    this.polarizationLeftSource = null;
    this.polarizationRightSource = null;

    this.polarizationCenterGain = null;
    this.polarizationLeftGain = null;
    this.polarizationRightGain = null;

    this.polarizationCenterFilters = [];
    this.polarizationLeftFilters = [];
    this.polarizationRightFilters = [];

    this.polarizationStrength = 0.0;


    // ---------------------------------------------------------
    // EQ settings
    // ---------------------------------------------------------

    this.filterCount = 5;

    this.temperatureEQSettings = [
        { frequency: 220,  gain: 12.0, q: 2.3 },
        { frequency: 537,  gain: 8.6,  q: 5.1 },
        { frequency: 810,  gain: 8.4,  q: 7.4 },
        { frequency: 1120, gain: 5.2,  q: 9.3 },
        { frequency: 1440, gain: 3.5,  q: 11.0 }
    ];

    this.polarizationEQSettings = [
        { frequency: 1000, gain: 0, q: 1 },
        { frequency: 1000, gain: 0, q: 1 },
        { frequency: 1000, gain: 0, q: 1 },
        { frequency: 1000, gain: 0, q: 1 },
        { frequency: 1000, gain: 0, q: 1 }
    ];

    this.polarizationReady = false;


    // =========================================================
    // NOISE GENERATORS
    // =========================================================

    this.createPinkNoiseBuffer = function() {

        let bufferSize =
            2 * this.context.sampleRate;

        let buffer =
            this.context.createBuffer(
                1,
                bufferSize,
                this.context.sampleRate
            );

        let output =
            buffer.getChannelData(0);

        let b0 = 0;
        let b1 = 0;
        let b2 = 0;
        let b3 = 0;
        let b4 = 0;
        let b5 = 0;
        let b6 = 0;

        for (
            let i = 0;
            i < bufferSize;
            i++
        ) {

            let white =
                Math.random() * 2 - 1;

            b0 =
                0.99886 * b0 +
                white * 0.0555179;

            b1 =
                0.99332 * b1 +
                white * 0.0750759;

            b2 =
                0.96900 * b2 +
                white * 0.1538520;

            b3 =
                0.86650 * b3 +
                white * 0.3104856;

            b4 =
                0.55000 * b4 +
                white * 0.5329522;

            b5 =
                -0.7616 * b5 -
                white * 0.0168980;

            output[i] =
                b0 +
                b1 +
                b2 +
                b3 +
                b4 +
                b5 +
                b6 +
                white * 0.5362;

            output[i] *= 0.08;

            b6 =
                white * 0.115926;
        }

        return buffer;
    };


    this.createBlueNoiseBuffer = function() {

        let bufferSize =
            2 * this.context.sampleRate;

        let buffer =
            this.context.createBuffer(
                1,
                bufferSize,
                this.context.sampleRate
            );

        let output =
            buffer.getChannelData(0);

        let previousWhite = 0;

        for (
            let i = 0;
            i < bufferSize;
            i++
        ) {

            let white =
                Math.random() * 2 - 1;

            output[i] =
                (
                    white -
                    previousWhite
                ) *
                0.08;

            previousWhite =
                white;
        }

        return buffer;
    };


    // =========================================================
    // EQ HELPERS
    // =========================================================

    this.normalizedEQSettings = function(
        settings
    ) {

        var normalized = [];

        settings =
            settings || [];

        for (
            var i = 0;
            i < this.filterCount;
            i++
        ) {

            if (
                i < settings.length
            ) {

                normalized.push({
                    frequency:
                        settings[i].frequency,

                    gain:
                        settings[i].gain,

                    q:
                        settings[i].q
                });

            } else {

                normalized.push({
                    frequency: 1000,
                    gain: 0,
                    q: 1
                });
            }
        }

        return normalized;
    };


    this.createEQChain = function(
        source,
        cents,
        filterStore,
        settings
    ) {

        var previous =
            source;

        var pitchRatio =
            Math.pow(
                2,
                cents / 1200
            );

        settings =
            this.normalizedEQSettings(
                settings
            );

        for (
            var i = 0;
            i < this.filterCount;
            i++
        ) {

            var setting =
                settings[i];

            var filter =
                this.context.createBiquadFilter();

            filter.type =
                "peaking";

            filter.frequency.value =
                setting.frequency *
                pitchRatio;

            filter.gain.value =
                setting.gain;

            filter.Q.value =
                setting.q;

            previous.connect(
                filter
            );

            previous =
                filter;

            filterStore.push(
                filter
            );
        }

        return previous;
    };


    this.updateFilterBank = function(
        filters,
        settings,
        cents
    ) {

        if (
            !this.context ||
            !filters ||
            filters.length === 0
        ) {
            return;
        }

        var normalized =
            this.normalizedEQSettings(
                settings
            );

        var now =
            this.context.currentTime;

        var rampTime =
            0.15;

        var pitchRatio =
            Math.pow(
                2,
                cents / 1200
            );

        for (
            var i = 0;
            i < filters.length;
            i++
        ) {

            var setting =
                normalized[i];

            filters[i].frequency
                .cancelScheduledValues(
                    now
                );

            filters[i].gain
                .cancelScheduledValues(
                    now
                );

            filters[i].Q
                .cancelScheduledValues(
                    now
                );

            filters[i].frequency
                .setTargetAtTime(
                    setting.frequency *
                        pitchRatio,
                    now,
                    rampTime
                );

            filters[i].gain
                .setTargetAtTime(
                    setting.gain,
                    now,
                    rampTime
                );

            filters[i].Q
                .setTargetAtTime(
                    setting.q,
                    now,
                    rampTime
                );
        }
    };


    // =========================================================
    // POLARIZATION VOICE BUILDER
    //
    // Same basic spectral-subtraction technique as Temperature:
    //
    // source -------- dry --------------------+
    //    \                                      \
    //     +---- EQ ---- inverted (-1) ----------+--> sum
    //
    // Then:
    //
    // sum -> delay -> pan -> gain -> polarization bus
    // =========================================================

    this.createPolarizationVoice = function(
        buffer,
        cents,
        delaySeconds,
        panValue,
        filterStore
    ) {

        var source =
            this.context.createBufferSource();

        source.buffer =
            buffer;

        source.loop =
            true;

        source.detune.value =
            0;


        var dryGain =
            this.context.createGain();

        dryGain.gain.value =
            1.0;


        var invertedGain =
            this.context.createGain();

        invertedGain.gain.value =
            -1.0;


        var sumGain =
            this.context.createGain();

        sumGain.gain.value =
            1.0;


        // Dry branch
        source.connect(
            dryGain
        );

        dryGain.connect(
            sumGain
        );


        // EQ-shaped inverted branch
        var filteredOutput =
            this.createEQChain(
                source,
                cents,
                filterStore,
                this.polarizationEQSettings
            );

        filteredOutput.connect(
            invertedGain
        );

        invertedGain.connect(
            sumGain
        );


        // Delay
        var delay =
            this.context.createDelay(
                0.1
            );

        delay.delayTime.value =
            delaySeconds;


        // Stereo pan
        var panner =
            this.context.createStereoPanner();

        panner.pan.value =
            panValue;


        // Voice level
        var outputGain =
            this.context.createGain();


        sumGain.connect(
            delay
        );

        delay.connect(
            panner
        );

        panner.connect(
            outputGain
        );

        outputGain.connect(
            this.polarizationBus
        );


        return {
            source:
                source,

            outputGain:
                outputGain
        };
    };


    // =========================================================
    // START AUDIO
    // =========================================================

    this.startTone = function() {

        if (
            this.context
        ) {
            return;
        }


        this.context =
            new (
                window.AudioContext ||
                window.webkitAudioContext
            )();


        // -----------------------------------------------------
        // Master output
        // -----------------------------------------------------

        this.masterGain =
            this.context.createGain();

        this.masterGain.gain.value =
            this.masterVolume;

        this.masterGain.connect(
            this.context.destination
        );


        // -----------------------------------------------------
        // Temperature / Polarization buses
        // -----------------------------------------------------

        this.temperatureBus =
            this.context.createGain();

        this.polarizationBus =
            this.context.createGain();


        this.temperatureBus.gain.value =
            (
                this.observationMode ===
                "temperature"
            )
                ? 1.0
                : 0.0;


        this.polarizationBus.gain.value =
            (
                this.observationMode ===
                "polarization"
            )
                ? 1.0
                : 0.0;


        this.temperatureBus.connect(
            this.masterGain
        );

        this.polarizationBus.connect(
            this.masterGain
        );


        // -----------------------------------------------------
        // One shared pink-noise realization
        // -----------------------------------------------------

        var pinkNoiseBuffer =
            this.createPinkNoiseBuffer();


        // =====================================================
        // TEMPERATURE
        // =====================================================

        this.noiseSource =
            this.context.createBufferSource();

        this.noiseSource.buffer =
            pinkNoiseBuffer;

        this.noiseSource.loop =
            true;


        this.dryGain =
            this.context.createGain();

        this.dryGain.gain.value =
            1.0;


        this.invertedGain =
            this.context.createGain();

        this.invertedGain.gain.value =
            -1.0;


        this.filters =
            [];


        // Dry pink noise
        this.noiseSource.connect(
            this.dryGain
        );

        this.dryGain.connect(
            this.temperatureBus
        );


        // EQ-shaped pink noise, polarity inverted
        var temperatureFilteredOutput =
            this.createEQChain(
                this.noiseSource,
                0,
                this.filters,
                this.temperatureEQSettings
            );

        temperatureFilteredOutput.connect(
            this.invertedGain
        );

        this.invertedGain.connect(
            this.temperatureBus
        );


        // Existing temperature "hotness" layer
        this.blueNoiseSource =
            this.context.createBufferSource();

        this.blueNoiseSource.buffer =
            this.createBlueNoiseBuffer();

        this.blueNoiseSource.loop =
            true;


        this.blueNoiseGain =
            this.context.createGain();

        this.blueNoiseGain.gain.value =
            0.0;


        this.blueNoiseSource.connect(
            this.blueNoiseGain
        );

        this.blueNoiseGain.connect(
            this.temperatureBus
        );


        // =====================================================
        // POLARIZATION
        //
        // Relative timing:
        //
        // Right   = 0 ms
        // Center  = 10 ms
        // Left    = 20 ms
        //
        // Therefore relative to center:
        //
        // Right   = -10 ms
        // Left    = +10 ms
        // =====================================================

        this.polarizationCenterFilters =
            [];

        this.polarizationLeftFilters =
            [];

        this.polarizationRightFilters =
            [];


        var centerVoice =
            this.createPolarizationVoice(
                pinkNoiseBuffer,
                0,
                0.010,
                0,
                this.polarizationCenterFilters
            );


        var leftVoice =
            this.createPolarizationVoice(
                pinkNoiseBuffer,
                -6,
                0.020,
                -1,
                this.polarizationLeftFilters
            );


        var rightVoice =
            this.createPolarizationVoice(
                pinkNoiseBuffer,
                6,
                0.000,
                1,
                this.polarizationRightFilters
            );


        this.polarizationCenterSource =
            centerVoice.source;

        this.polarizationLeftSource =
            leftVoice.source;

        this.polarizationRightSource =
            rightVoice.source;


        this.polarizationCenterGain =
            centerVoice.outputGain;

        this.polarizationLeftGain =
            leftVoice.outputGain;

        this.polarizationRightGain =
            rightVoice.outputGain;


        // -----------------------------------------------------
        // Initial polarization width
        // -----------------------------------------------------

        var initialAngle =
            this.polarizationStrength *
            Math.PI /
            2;


        var initialCenter =
            Math.cos(
                initialAngle
            );


        var initialSide =
            Math.sin(
                initialAngle
            ) /
            Math.sqrt(2);


        this.polarizationCenterGain.gain.value =
            initialCenter;

        this.polarizationLeftGain.gain.value =
            initialSide;

        this.polarizationRightGain.gain.value =
            initialSide;


        // -----------------------------------------------------
        // Start all sources together
        // -----------------------------------------------------

        var startTime =
            this.context.currentTime +
            0.05;


        this.noiseSource.start(
            startTime
        );

        this.blueNoiseSource.start(
            startTime
        );

        this.polarizationCenterSource.start(
            startTime
        );

        this.polarizationLeftSource.start(
            startTime
        );

        this.polarizationRightSource.start(
            startTime
        );


        this.setHotness(
            this.hotness
        );
    };


    // =========================================================
    // STOP AUDIO
    // =========================================================

    this.stopTone = function() {

        var sources = [
            this.noiseSource,
            this.blueNoiseSource,
            this.polarizationCenterSource,
            this.polarizationLeftSource,
            this.polarizationRightSource
        ];


        for (
            var i = 0;
            i < sources.length;
            i++
        ) {

            if (
                sources[i]
            ) {

                try {

                    sources[i].stop();

                } catch (e) {

                    // Source may already have stopped.
                }
            }
        }


        this.noiseSource =
            null;

        this.blueNoiseSource =
            null;

        this.blueNoiseGain =
            null;


        this.dryGain =
            null;

        this.invertedGain =
            null;

        this.filters =
            [];


        this.polarizationCenterSource =
            null;

        this.polarizationLeftSource =
            null;

        this.polarizationRightSource =
            null;


        this.polarizationCenterGain =
            null;

        this.polarizationLeftGain =
            null;

        this.polarizationRightGain =
            null;


        this.polarizationCenterFilters =
            [];

        this.polarizationLeftFilters =
            [];

        this.polarizationRightFilters =
            [];


        this.temperatureBus =
            null;

        this.polarizationBus =
            null;

        this.masterGain =
            null;

        this.context =
            null;
    };


    // =========================================================
    // SPECTRUM -> AUDIO EQ
    // =========================================================

    this.setSpectrumEQ = function(
        settings
    ) {

        if (
            this.observationMode ===
            "polarization"
        ) {

            this.polarizationEQSettings =
                this.normalizedEQSettings(
                    settings
                );

            this.updateFilterBank(
                this.polarizationCenterFilters,
                this.polarizationEQSettings,
                0
            );

            this.updateFilterBank(
                this.polarizationLeftFilters,
                this.polarizationEQSettings,
                -6
            );

            this.updateFilterBank(
                this.polarizationRightFilters,
                this.polarizationEQSettings,
                6
            );

            this.polarizationReady =
                true;

            /*
            * If Polarization is currently selected,
            * bring its bus up only now that valid EE
            * settings have arrived.
            */
            if (
                this.context &&
                this.polarizationBus
            ) {

                var now =
                    this.context.currentTime;

                this.polarizationBus.gain
                    .cancelScheduledValues(
                        now
                    );

                this.polarizationBus.gain
                    .setTargetAtTime(
                        1.0,
                        now,
                        0.05
                    );
            }

            return;
        }


        this.temperatureEQSettings =
            this.normalizedEQSettings(
                settings
            );

        this.updateFilterBank(
            this.filters,
            this.temperatureEQSettings,
            0
        );
    };


    // =========================================================
    // TEMPERATURE HOTNESS
    // =========================================================

    this.setHotness = function(
        hotness
    ) {

        this.hotness =
            Math.max(
                0,
                Math.min(
                    1,
                    hotness
                )
            );


        if (
            !this.context ||
            !this.blueNoiseGain
        ) {
            return;
        }


        var now =
            this.context.currentTime;

        var rampTime =
            0.2;


        var blueLevel =
            0.18 *
            Math.max(
                0,
                this.hotness -
                0.5
            ) *
            2;


        blueLevel =
            Math.max(
                0,
                Math.min(
                    0.18,
                    blueLevel
                )
            );


        this.blueNoiseGain.gain
            .cancelScheduledValues(
                now
            );


        this.blueNoiseGain.gain
            .setTargetAtTime(
                blueLevel,
                now,
                rampTime
            );
    };


    this.testHotness = function() {

        this.setHotness(
            1.0
        );
    };


    this.resetHotness = function() {

        this.setHotness(
            0.5
        );
    };


    // =========================================================
    // POLARIZATION STRENGTH -> STEREO WIDTH
    // =========================================================

    this.setPolarizationStrength = function(
        strength
    ) {

        this.polarizationStrength =
            Math.max(
                0,
                Math.min(
                    1,
                    strength
                )
            );


        if (
            !this.context ||
            !this.polarizationCenterGain ||
            !this.polarizationLeftGain ||
            !this.polarizationRightGain
        ) {
            return;
        }


        var now =
            this.context.currentTime;

        var rampTime =
            0.20;


        var angle =
            this.polarizationStrength *
            Math.PI /
            2;


        var center =
            Math.cos(
                angle
            );


        var side =
            Math.sin(
                angle
            ) /
            Math.sqrt(2);


        this.polarizationCenterGain.gain
            .cancelScheduledValues(
                now
            );

        this.polarizationLeftGain.gain
            .cancelScheduledValues(
                now
            );

        this.polarizationRightGain.gain
            .cancelScheduledValues(
                now
            );


        this.polarizationCenterGain.gain
            .setTargetAtTime(
                center,
                now,
                rampTime
            );


        this.polarizationLeftGain.gain
            .setTargetAtTime(
                side,
                now,
                rampTime
            );


        this.polarizationRightGain.gain
            .setTargetAtTime(
                side,
                now,
                rampTime
            );
    };


    // =========================================================
    // TEMPERATURE / POLARIZATION AUDIO MODE
    // =========================================================

    this.setObservationMode = function(
        mode
    ) {

        if (
            mode !== "temperature" &&
            mode !== "polarization"
        ) {
            return;
        }


        this.observationMode =
            mode;

        if (
            mode ===
            "polarization"
        ) {

            this.polarizationReady =
                false;
        }

        if (
            !this.context ||
            !this.temperatureBus ||
            !this.polarizationBus
        ) {
            return;
        }


        var now =
            this.context.currentTime;

        var rampTime =
            0.05;


        var temperatureLevel =
            (
                mode ===
                "temperature"
            )
                ? 1.0
                : 0.0;


        var polarizationLevel =
            (
                mode ===
                    "polarization" &&
                this.polarizationReady
            )
                ? 1.0
                : 0.0;


        this.temperatureBus.gain
            .cancelScheduledValues(
                now
            );


        this.polarizationBus.gain
            .cancelScheduledValues(
                now
            );


        this.temperatureBus.gain
            .setTargetAtTime(
                temperatureLevel,
                now,
                rampTime
            );


        this.polarizationBus.gain
            .setTargetAtTime(
                polarizationLevel,
                now,
                rampTime
            );
    };


    // =========================================================
    // TEST / RESET EQ
    // =========================================================

    this.testEQ = function() {

        this.setSpectrumEQ([
            {
                frequency: 220,
                gain: 20,
                q: 2
            },
            {
                frequency: 537,
                gain: 0,
                q: 5
            },
            {
                frequency: 810,
                gain: 0,
                q: 5
            },
            {
                frequency: 1120,
                gain: 0,
                q: 5
            },
            {
                frequency: 1440,
                gain: 0,
                q: 5
            }
        ]);
    };


    this.resetEQ = function() {

        this.setSpectrumEQ([
            {
                frequency: 220,
                gain: 12.0,
                q: 2.3
            },
            {
                frequency: 537,
                gain: 8.6,
                q: 5.1
            },
            {
                frequency: 810,
                gain: 8.4,
                q: 7.4
            },
            {
                frequency: 1120,
                gain: 5.2,
                q: 9.3
            },
            {
                frequency: 1440,
                gain: 3.5,
                q: 11.0
            }
        ]);
    };


    // =========================================================
    // MASTER VOLUME
    // =========================================================

    this.setVolume = function(
        value
    ) {

        value =
            Math.max(
                0,
                Math.min(
                    1,
                    value
                )
            );


        this.masterVolume =
            value;


        if (
            this.masterGain
        ) {

            this.masterGain.gain.value =
                this.masterVolume;
        }
    };


    /*
     * Retained for compatibility with simulator.js.
     */
    this.setAmplitudes = function(
        amplitudes
    ) {};
}