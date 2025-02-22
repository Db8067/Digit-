document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');
    const predictedDigitSpan = document.getElementById('predictedDigit');
    const confidenceSpan = document.getElementById('confidence');
    const canvasOverlay = document.querySelector('.canvas-overlay');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 2000; // 2 seconds

    // Setup canvas
    ctx.lineWidth = 20;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);

        // Start new path for smoother lines
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(lastX, lastY);
        ctx.stroke();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const [currentX, currentY] = getCoordinates(e);

        // Draw curved line for smoother appearance
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.quadraticCurveTo(lastX, lastY, (lastX + currentX) / 2, (lastY + currentY) / 2);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath(); // Reset the path
    }

    function getCoordinates(e) {
        if (e.type.includes('touch')) {
            const rect = canvas.getBoundingClientRect();
            const touch = e.touches[0];
            return [
                touch.clientX - rect.left,
                touch.clientY - rect.top
            ];
        }
        return [e.offsetX, e.offsetY];
    }

    function handleTouchStart(e) {
        e.preventDefault();
        startDrawing(e);
    }

    function handleTouchMove(e) {
        e.preventDefault();
        draw(e);
    }

    // Clear canvas with animation
    clearBtn.addEventListener('click', function() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.classList.add('d-none');
        errorDiv.classList.add('d-none');

        // Add visual feedback
        clearBtn.classList.add('btn-danger');
        setTimeout(() => clearBtn.classList.remove('btn-danger'), 200);
    });

    async function predictWithRetry() {
        try {
            const imageData = canvas.toDataURL('image/png');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });

            const data = await response.json();

            if (response.status === 503 && retryCount < MAX_RETRIES) {
                // Model is still training, retry after delay
                retryCount++;
                errorDiv.innerHTML = `<i class="bi bi-info-circle"></i> Model is still initializing... Retrying in ${RETRY_DELAY/1000} seconds (Attempt ${retryCount}/${MAX_RETRIES})`;
                errorDiv.classList.remove('d-none');
                setTimeout(predictWithRetry, RETRY_DELAY);
                return;
            }

            if (data.success) {
                predictedDigitSpan.textContent = data.digit;

                // Animate confidence counter
                const targetConfidence = data.confidence;
                const startConfidence = 0;
                const duration = 1000;
                const startTime = performance.now();

                function updateConfidence(currentTime) {
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);

                    const currentConfidence = Math.round(startConfidence + (targetConfidence - startConfidence) * progress);
                    confidenceSpan.textContent = currentConfidence;

                    if (progress < 1) {
                        requestAnimationFrame(updateConfidence);
                    }
                }

                requestAnimationFrame(updateConfidence);
                resultDiv.classList.remove('d-none');
                errorDiv.classList.add('d-none');
            } else {
                errorDiv.innerHTML = `<i class="bi bi-exclamation-triangle"></i> ${data.error || 'An error occurred during prediction. Please try again.'}`;
                errorDiv.classList.remove('d-none');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            errorDiv.innerHTML = '<i class="bi bi-exclamation-triangle"></i> Network error. Please check your connection and try again.';
            errorDiv.classList.remove('d-none');
        } finally {
            loadingDiv.classList.add('d-none');
            canvasOverlay.classList.add('d-none');
            retryCount = 0; // Reset retry counter
        }
    }

    // Predict with improved feedback
    predictBtn.addEventListener('click', function() {
        resultDiv.classList.add('d-none');
        errorDiv.classList.add('d-none');
        loadingDiv.classList.remove('d-none');
        canvasOverlay.classList.remove('d-none');
        retryCount = 0;
        predictWithRetry();
    });
});