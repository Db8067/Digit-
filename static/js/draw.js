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

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Setup canvas
    ctx.lineWidth = 15;
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
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDrawing);

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const [currentX, currentY] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getCoordinates(e) {
        if (e.type.includes('touch')) {
            const rect = canvas.getBoundingClientRect();
            return [
                e.touches[0].clientX - rect.left,
                e.touches[0].clientY - rect.top
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

    // Clear canvas
    clearBtn.addEventListener('click', function() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.classList.add('d-none');
        errorDiv.classList.add('d-none');
    });

    // Predict
    predictBtn.addEventListener('click', async function() {
        resultDiv.classList.add('d-none');
        errorDiv.classList.add('d-none');
        loadingDiv.classList.remove('d-none');

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

            if (data.success) {
                predictedDigitSpan.textContent = data.digit;
                confidenceSpan.textContent = data.confidence;
                resultDiv.classList.remove('d-none');
            } else {
                errorDiv.classList.remove('d-none');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            errorDiv.classList.remove('d-none');
        } finally {
            loadingDiv.classList.add('d-none');
        }
    });
});
