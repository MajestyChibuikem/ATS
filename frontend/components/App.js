class App {
    constructor() {
        this.container = null;
        this.currentComponent = null;
        this.components = {
            login: null,
            dashboard: null
        };
        console.log('App constructor initialized');
    }

    init(container) {
        console.log('App init called with container:', container);
        this.container = container;
        this.setupRouting();
        this.handleRoute();
    }

    setupRouting() {
        console.log('Setting up routing...');
        
        // Handle initial route
        window.addEventListener('load', () => {
            console.log('Window loaded, handling route...');
            this.handleRoute();
        });

        // Handle hash changes
        window.addEventListener('hashchange', () => {
            console.log('Hash changed, handling route...');
            this.handleRoute();
        });
    }

    handleRoute() {
        const hash = window.location.hash.slice(1) || 'login';
        const isAuthenticated = this.checkAuthentication();
        
        console.log('Handling route:', { hash, isAuthenticated });

        if (hash === 'login' || !isAuthenticated) {
            console.log('Showing login page');
            this.showLogin();
        } else {
            console.log('Showing dashboard');
            this.showDashboard();
        }
    }

    checkAuthentication() {
        const token = localStorage.getItem('authToken');
        const hasToken = !!token;
        console.log('Checking authentication:', { hasToken, token: token ? 'present' : 'none' });
        return hasToken;
    }

    showLogin() {
        console.log('Creating/showing login component');
        if (!this.components.login) {
            console.log('Creating new Login component');
            this.components.login = new Login();
        }
        
        this.currentComponent = this.components.login;
        this.components.login.mount(this.container);
        console.log('Login component mounted');
    }

    showDashboard() {
        console.log('Creating/showing dashboard component');
        if (!this.components.dashboard) {
            console.log('Creating new Dashboard component');
            this.components.dashboard = new Dashboard();
        }
        
        this.currentComponent = this.components.dashboard;
        this.components.dashboard.mount(this.container);
        console.log('Dashboard component mounted');
    }

    logout() {
        console.log('Logging out...');
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
        window.location.hash = '#login';
    }
}
