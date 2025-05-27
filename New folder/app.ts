// TypeScript interface definitions
interface DetectionStatus {
  status: string;
  ear?: number;
  mar?: number;
  isAlert?: boolean;
  alertMessage?: string;
}

class DrowsinessDetectionApp {
  private statusElement: HTMLElement;
  private videoElement: HTMLVideoElement | null;
  private earValueElement: HTMLElement;
  private marValueElement: HTMLElement;
  private alertElement: HTMLElement;
  private alertSound: HTMLAudioElement;
  private statusUpdateInterval: number;
  private isRunning: boolean = false;
  
  constructor() {
    this.statusElement = document.getElementById('status') as HTMLElement;
    this.videoElement = document.getElementById('videoFeed') as HTMLVideoElement;
    this.earValueElement = document.getElementById('earValue') as HTMLElement;
    this.marValueElement = document.getElementById('marValue') as HTMLElement;
    this.alertElement = document.getElementById('alertMessage') as HTMLElement;
    
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
  }
  
  private async startDetection(): Promise<void> {
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
  
  private async stopDetection(): Promise<void> {
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
  
  private async updateStatus(): Promise<void> {
    try {
      const response = await fetch('/status');
      const data: DetectionStatus = await response.json();
      
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
  
  private updateUIState(): void {
    const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
    const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
    const statusContainer = document.getElementById('statusContainer') as HTMLElement;
    
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
  
  private showAlert(message: string): void {
    this.alertElement.textContent = message;
    this.alertElement.classList.add('visible');
    document.body.classList.add('alert-active');
  }
  
  private clearAlert(): void {
    this.alertElement.classList.remove('visible');
    document.body.classList.remove('alert-active');
  }
  
  private playAlertSound(): void {
    this.alertSound.play().catch(err => console.error('Failed to play alert sound:', err));
  }
  
  private logout(): void {
    window.location.href = '/login.html';
  }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new DrowsinessDetectionApp();
});