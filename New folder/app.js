// JavaScript implementation of the TypeScript code

class DrowsinessDetectionApp {
    constructor() {
        this.statusElement = document.getElementById('statusValue');
        this.videoElement = document.getElementById('videoFeed');
        this.earValueElement = document.getElementById('earValue');
        this.marValueElement = document.getElementById('marValue');
        this.alertElement = document.getElementById('alertMessage');
        
        // Create alert sound
        this.alertSound = new Audio();
        this.alertSound.src = 'https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3';
        this.alertSound.load();
        
        // Set up event listeners
        document.getElementById('startBtn')?.addEventListener('click', () => this.startDetection());
        document.getElementById('stopBtn')?.addEventListener('click', () => this.stopDetection());
        document.getElementById('logoutBtn')?.addEventListener('click', () => this.logout());
        
        // Initialize status updates
        this.statusUpdateInterval = window.setInterval(() => this.updateStatus(), 1000);
        this.isRunning = false;
    }
    
    async startDetection() {
        try {
            const response = await fetch('/start_detection', {method: 'POST'});
            const data = await response.json();
            this.isRunning = true;
            this.updateUIState();
        } catch (error) {
            console.error('Failed to start detection:', error);
            this.showAlert('Failed to start detection. Please try again.');
        }
    }
    
    async stopDetection() {
        try {
            const response = await fetch('/stop_detection', {method: 'POST'});
            const data = await response.json();
            this.isRunning = false;
            this.updateUIState();
        } catch (error) {
            console.error('Failed to stop detection:', error);
            this.showAlert('Failed to stop detection. Please try again.');
        }
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            // Update status display
            this.statusElement.textContent = data.status;
            
            // Update EAR/MAR values if available
            if (data.ear !== undefined) {
                this.earValueElement.textContent = data.ear.toFixed(2);
            }
            
            if (data.mar !== undefined) {
                this.marValueElement.textContent = data.mar.toFixed(2);
            }
            
            // Handle alerts
            if (data.isAlert) {
                this.showAlert(data.alertMessage || 'Drowsiness detected!');
                this.playAlertSound();
            } else {
                this.clearAlert();
            }
            
            this.updateUIState();
        } catch (error) {
            console.error('Failed to update status:', error);
        }
    }
    
    updateUIState() {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusContainer = document.getElementById('statusContainer');
        
        if (this.isRunning) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusContainer.classList.add('active');
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusContainer.classList.remove('active');
        }
    }
    
    showAlert(message) {
        this.alertElement.textContent = message;
        this.alertElement.classList.add('visible');
        document.body.classList.add('alert-active');
    }
    
    clearAlert() {
        this.alertElement.classList.remove('visible');
        document.body.classList.remove('alert-active');
    }
    
    playAlertSound() {
        this.alertSound.play().catch(err => console.error('Failed to play alert sound:', err));
    }
    
    logout() {
        window.location.href = '/login.html';
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DrowsinessDetectionApp();
});

// Add recording functionality
const startRecordingBtn = document.getElementById('startRecordingBtn');
const stopRecordingBtn = document.getElementById('stopRecordingBtn');
let isRecording = false;

startRecordingBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/start_recording', {method: 'POST'});
        const data = await response.json();
        
        if (data.status === 'recording_started' || data.status === 'already_recording') {
            isRecording = true;
            startRecordingBtn.disabled = true;
            stopRecordingBtn.disabled = false;
            console.log('Recording started:', data.filename);
        } else {
            console.error('Failed to start recording:', data.message);
            alert('Failed to start recording. Please check console for details.');
        }
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Error starting recording. Please try again.');
    }
});

stopRecordingBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/stop_recording', {method: 'POST'});
        const data = await response.json();
        
        if (data.status === 'recording_stopped') {
            isRecording = false;
            startRecordingBtn.disabled = false;
            stopRecordingBtn.disabled = true;
            console.log('Recording stopped. Duration:', data.duration, 'seconds');
            alert(`Recording stopped and saved. Duration: ${Math.floor(data.duration / 60)}:${Math.floor(data.duration % 60).toString().padStart(2, '0')}`);
        } else {
            console.error('Failed to stop recording:', data.status);
        }
    } catch (error) {
        console.error('Error stopping recording:', error);
        alert('Error stopping recording. Please try again.');
    }
});