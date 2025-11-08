class Login {
    constructor() {
        this.container = null;
    }

    render() {
        return `
            <div class="login-container">
                <div class="login-card">
                    <h1 class="login-title">Welcome to ANITA</h1>
                    <p class="login-subtitle">Smart Student Analytics System</p>
                    
                    <form id="loginForm" class="login-form">
                        <div class="form-group">
                            <label for="username" class="form-label">Username</label>
                            <input 
                                type="text" 
                                id="username" 
                                class="form-input" 
                                placeholder="Enter your username"
                                value="admin"
                                required
                            >
                        </div>
                        
                        <div class="form-group">
                            <label for="password" class="form-label">Password</label>
                            <input 
                                type="password" 
                                id="password" 
                                class="form-input" 
                                placeholder="Enter your password"
                                value="admin123"
                                required
                            >
                        </div>
                        
                        <button type="submit" class="login-button">
                            <span class="button-text">Sign In</span>
                            <span class="loading-spinner" id="loginSpinner" style="display: none;">
                                <i class="fas fa-spinner fa-spin"></i>
                            </span>
                        </button>
                    </form>
                    
                    <div class="login-footer">
                        <p class="footer-text">
                            Don't have an account? 
                            <a href="#" class="signin-link" id="showRegister">Register</a>
                        </p>
                    </div>
                    
                    <div class="debug-info">
                        <strong>Debug Info:</strong><br>
                        <span id="debugStatus">Ready to login...</span>
                    </div>
                </div>
            </div>
        `;
    }

    mount(container) {
        this.container = container;
        container.innerHTML = this.render();
        this.attachEventListeners();
        console.log('Login component mounted');
    }

    attachEventListeners() {
        const loginForm = document.getElementById('loginForm');
        const showRegisterLink = document.getElementById('showRegister');

        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                console.log('Login form submitted');
                this.handleLogin();
            });
        } else {
            console.error('Login form not found');
        }

        if (showRegisterLink) {
            showRegisterLink.addEventListener('click', (e) => {
                e.preventDefault();
                this.showRegister();
            });
        }
    }

    async handleLogin() {
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const spinner = document.getElementById('loginSpinner');
        const buttonText = document.querySelector('.button-text');
        const debugStatus = document.getElementById('debugStatus');

        console.log('Attempting login with:', { username, password: '***' });

        // Show loading state
        if (spinner) spinner.style.display = 'inline-block';
        if (buttonText) buttonText.style.display = 'none';
        if (debugStatus) debugStatus.textContent = 'Logging in...';

        try {
            console.log('Making API call to login endpoint...');
            
            const response = await fetch('http://localhost:8000/api/v1/auth/login/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    username: username,
                    password: password 
                })
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);

            if (response.ok) {
                const data = await response.json();
                console.log('Login successful:', data);
                
                localStorage.setItem('authToken', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));
                
                if (debugStatus) debugStatus.textContent = 'Login successful! Redirecting...';
                
                // Navigate to dashboard
                window.location.hash = '#dashboard';
                this.showSuccessMessage('Login successful!');
            } else {
                const errorData = await response.json();
                console.error('Login failed:', errorData);
                if (debugStatus) debugStatus.textContent = `Login failed: ${errorData.error || 'Unknown error'}`;
                this.showErrorMessage(errorData.error || 'Login failed. Please check your credentials.');
            }
        } catch (error) {
            console.error('Login error:', error);
            if (debugStatus) debugStatus.textContent = `Connection error: ${error.message}`;
            this.showErrorMessage('Connection error. Please try again.');
        } finally {
            // Hide loading state
            if (spinner) spinner.style.display = 'none';
            if (buttonText) buttonText.style.display = 'inline';
        }
    }

    showRegister() {
        // For now, just show a message. We can implement register later
        this.showInfoMessage('Registration feature coming soon!');
    }

    showSuccessMessage(message) {
        this.showMessage(message, 'success');
    }

    showErrorMessage(message) {
        this.showMessage(message, 'error');
    }

    showInfoMessage(message) {
        this.showMessage(message, 'info');
    }

    showMessage(message, type) {
        console.log(`Showing ${type} message:`, message);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type}`;
        messageDiv.textContent = message;
        
        this.container.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }
}
