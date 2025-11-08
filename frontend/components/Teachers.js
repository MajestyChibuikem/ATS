class Teachers {
    constructor() {
        this.container = null;
        this.teachers = [];
        this.filteredTeachers = [];
        this.currentFilters = {
            search: '',
            specialization: '',
            qualification: '',
            experience: ''
        };
        this.cacheKey = 'teachers_data_cache';
        this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
        this.isLoading = false; // Prevent multiple simultaneous loads
    }

    render() {
        return `
            <div class="teachers-container">
                <!-- Header -->
                <div class="page-header">
                    <h1 class="page-title">Teacher List</h1>
                </div>

                <!-- Search and Filters -->
                <div class="search-filters-section">
                    <div class="search-bar">
                        <div class="search-input-wrapper">
                            <i class="fas fa-search search-icon"></i>
                            <input 
                                type="text" 
                                id="teacherSearch" 
                                class="search-input" 
                                placeholder="Search for teacher names"
                            >
                        </div>
                    </div>
                    
                    <div class="filters-row">
                        <div class="filter-group">
                            <label for="specializationFilter" class="filter-label">Specialization</label>
                            <select id="specializationFilter" class="filter-select">
                                <option value="">All Specializations</option>
                                <option value="Mathematics">Mathematics</option>
                                <option value="Sciences">Sciences</option>
                                <option value="Languages">Languages</option>
                                <option value="Arts">Arts</option>
                                <option value="General">General</option>
                            </select>
                        </div>
                        
                        <div class="filter-group">
                            <label for="qualificationFilter" class="filter-label">Qualification</label>
                            <select id="qualificationFilter" class="filter-select">
                                <option value="">All Qualifications</option>
                                <option value="HND + PGDE">HND + PGDE</option>
                                <option value="B.Sc + PGDE">B.Sc + PGDE</option>
                                <option value="B.Ed">B.Ed</option>
                                <option value="M.Ed">M.Ed</option>
                                <option value="PhD">PhD</option>
                            </select>
                        </div>
                        
                        <div class="filter-group">
                            <label for="experienceFilter" class="filter-label">Experience Level</label>
                            <select id="experienceFilter" class="filter-select">
                                <option value="">All Levels</option>
                                <option value="Novice">Novice (0-4 years)</option>
                                <option value="Intermediate">Intermediate (5-9 years)</option>
                                <option value="Experienced">Experienced (10-19 years)</option>
                                <option value="Senior">Senior (20+ years)</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="actions-row">
                        <button class="btn-primary" id="addTeacherBtn">
                            <i class="fas fa-plus"></i>
                            Add New Teacher
                        </button>
                        <button class="btn-secondary" id="refreshTeachersBtn">
                            <i class="fas fa-sync-alt"></i>
                            Refresh Data
                        </button>
                    </div>
                </div>

                <!-- Teachers Table -->
                <div class="table-container">
                    <table class="teachers-table">
                        <thead>
                            <tr>
                                <th>Teacher Name</th>
                                <th>Specialization</th>
                                <th>Qualification</th>
                                <th>Experience</th>
                                <th>Performance Rating</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="teachersTableBody">
                            <!-- Teachers will be populated here -->
                        </tbody>
                    </table>
                </div>

                <!-- Loading State -->
                <div id="loadingTeachersState" class="loading-state" style="display: none;">
                    <div class="loading-spinner">
                        <i class="fas fa-spinner fa-spin"></i>
                    </div>
                    <p>Loading teachers...</p>
                </div>

                <!-- Empty State -->
                <div id="emptyTeachersState" class="empty-state" style="display: none;">
                    <i class="fas fa-chalkboard-teacher"></i>
                    <h3>No Teachers Found</h3>
                    <p>No teachers match your current filters.</p>
                </div>

                <!-- Stats Bar -->
                <div class="stats-bar">
                    <div class="stat-item">
                        <span class="stat-label">Total Teachers:</span>
                        <span class="stat-value" id="totalTeachersCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Showing:</span>
                        <span class="stat-value" id="showingTeachersCount">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Last Updated:</span>
                        <span class="stat-value" id="lastTeachersUpdated">-</span>
                    </div>
                </div>
            </div>
        `;
    }

    mount(container) {
        this.container = container;
        container.innerHTML = this.render();
        this.attachEventListeners();
        
        // Test API call immediately
        this.testAPIConnection();
        
        this.loadTeachers();
    }

    async testAPIConnection() {
        console.log('Testing Teachers API connection...');
        const token = localStorage.getItem('authToken');
        console.log('Token available:', !!token);
        if (token) {
            console.log('Token starts with:', token.substring(0, 20));
        }
        
        try {
            const response = await fetch('http://localhost:8000/api/v1/teachers/', {
                method: 'GET',
                headers: {
                    'Authorization': `Token ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            console.log('Test Teachers API response status:', response.status);
            if (response.ok) {
                const data = await response.json();
                console.log('Test Teachers API success, teachers count:', data.length);
            } else {
                console.log('Test Teachers API failed with status:', response.status);
            }
        } catch (error) {
            console.error('Test Teachers API error:', error);
        }
    }

    attachEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('teacherSearch');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.currentFilters.search = e.target.value;
                this.filterTeachers();
            });
        }

        // Filter functionality
        const specializationFilter = document.getElementById('specializationFilter');
        const qualificationFilter = document.getElementById('qualificationFilter');
        const experienceFilter = document.getElementById('experienceFilter');

        if (specializationFilter) {
            specializationFilter.addEventListener('change', (e) => {
                this.currentFilters.specialization = e.target.value;
                this.filterTeachers();
            });
        }

        if (qualificationFilter) {
            qualificationFilter.addEventListener('change', (e) => {
                this.currentFilters.qualification = e.target.value;
                this.filterTeachers();
            });
        }

        if (experienceFilter) {
            experienceFilter.addEventListener('change', (e) => {
                this.currentFilters.experience = e.target.value;
                this.filterTeachers();
            });
        }

        // Add teacher button
        const addTeacherBtn = document.getElementById('addTeacherBtn');
        if (addTeacherBtn) {
            addTeacherBtn.addEventListener('click', () => {
                this.showAddTeacherModal();
            });
        }

        // Refresh button
        const refreshTeachersBtn = document.getElementById('refreshTeachersBtn');
        if (refreshTeachersBtn) {
            refreshTeachersBtn.addEventListener('click', () => {
                console.log('Refresh teachers button clicked');
                this.clearCache();
                this.loadTeachers(true); // Force refresh
            });
        }
    }

    async loadTeachers(forceRefresh = false) {
        // Prevent multiple simultaneous loads
        if (this.isLoading) {
            console.log('Already loading teachers, skipping...');
            return;
        }
        
        this.isLoading = true;
        
        const loadingState = document.getElementById('loadingTeachersState');
        const tableBody = document.getElementById('teachersTableBody');

        if (loadingState) loadingState.style.display = 'flex';
        if (tableBody) tableBody.innerHTML = '';

        try {
            // Check cache first (unless force refresh)
            if (!forceRefresh) {
                const cachedData = this.getCachedData();
                if (cachedData && cachedData.length > 0) {
                    console.log('Using cached data:', cachedData.length, 'teachers');
                    this.teachers = cachedData;
                    this.filteredTeachers = [...this.teachers];
                    this.renderTeachers();
                    this.updateStats();
                    if (loadingState) loadingState.style.display = 'none';
                    this.showMessage(`Loaded ${cachedData.length} teachers from cache`, 'success');
                    return;
                }
            }

            // Load from API only if no cached data or force refresh
            console.log('Loading teachers from API (forceRefresh:', forceRefresh, ')');
            const teachers = await this.loadTeachersFromAPI();
            console.log('API returned teachers:', teachers ? teachers.length : 0);
            
            if (teachers && teachers.length > 0) {
                console.log('Using API data:', teachers.length, 'teachers');
                this.teachers = teachers;
                this.cacheData(teachers);
                this.showMessage(`Loaded ${teachers.length} teachers from database`, 'success');
            } else {
                console.log('API returned no data, using sample data');
                // Fallback to sample data if API fails
                this.teachers = this.getSampleTeachers();
                this.showMessage('Using sample data (API unavailable)', 'info');
            }
        } catch (error) {
            console.error('Error loading teachers:', error);
            // Use sample data as fallback
            this.teachers = this.getSampleTeachers();
            this.showMessage('Error loading data, using sample data', 'error');
        }

        this.filteredTeachers = [...this.teachers];
        this.renderTeachers();
        this.updateStats();
        
        if (loadingState) loadingState.style.display = 'none';
        this.isLoading = false;
    }

    getCachedData() {
        const cached = localStorage.getItem(this.cacheKey);
        if (!cached) return null;

        try {
            const data = JSON.parse(cached);
            const now = Date.now();
            
            if (data.timestamp && (now - data.timestamp) < this.cacheExpiry) {
                return data.teachers;
            }
        } catch (error) {
            console.error('Error parsing cached data:', error);
        }
        
        return null;
    }

    cacheData(teachers) {
        try {
            const cacheData = {
                teachers: teachers,
                timestamp: Date.now()
            };
            localStorage.setItem(this.cacheKey, JSON.stringify(cacheData));
            console.log('Cached', teachers.length, 'teachers');
        } catch (error) {
            console.error('Error caching data:', error);
        }
    }

    clearCache() {
        try {
            localStorage.removeItem(this.cacheKey);
            console.log('Teachers cache cleared');
        } catch (error) {
            console.error('Error clearing cache:', error);
        }
    }

    async loadTeachersFromAPI() {
        const token = localStorage.getItem('authToken');
        console.log('Token from localStorage:', token ? token.substring(0, 20) + '...' : 'No token');
        
        if (!token) {
            console.error('No auth token found');
            return [];
        }

        try {
            console.log('Loading teachers from API...');
            console.log('API URL: http://localhost:8000/api/v1/teachers/');
            console.log('Token being used:', token.substring(0, 20) + '...');
            
            const response = await fetch('http://localhost:8000/api/v1/teachers/', {
                method: 'GET',
                headers: {
                    'Authorization': `Token ${token}`,
                    'Content-Type': 'application/json'
                }
            });

            console.log('API Response status:', response.status);
            console.log('API Response headers:', response.headers);

            if (response.ok) {
                const data = await response.json();
                console.log('API returned data:', data);
                console.log('Number of teachers from API:', data.length);
                return data || [];
            } else {
                const errorText = await response.text();
                console.error('API Error:', response.status, errorText);
                return [];
            }
        } catch (error) {
            console.error('API call failed:', error);
            console.error('Error details:', error.message);
            return [];
        }
    }

    getSampleTeachers() {
        return [
            {
                id: 1,
                name: 'Dr. Sarah Johnson',
                specialization: 'Mathematics',
                qualification: 'PhD',
                years_experience: 15,
                experience_level: 'Experienced',
                performance_rating: 4.5,
                teaching_load: 25,
                years_at_school: 8
            },
            {
                id: 2,
                name: 'Mr. Michael Chen',
                specialization: 'Sciences',
                qualification: 'M.Ed',
                years_experience: 8,
                experience_level: 'Intermediate',
                performance_rating: 4.2,
                teaching_load: 30,
                years_at_school: 5
            },
            {
                id: 3,
                name: 'Mrs. Patricia Williams',
                specialization: 'Languages',
                qualification: 'B.Ed',
                years_experience: 12,
                experience_level: 'Experienced',
                performance_rating: 4.8,
                teaching_load: 20,
                years_at_school: 10
            },
            {
                id: 4,
                name: 'Mr. David Rodriguez',
                specialization: 'Arts',
                qualification: 'B.Sc + PGDE',
                years_experience: 6,
                experience_level: 'Intermediate',
                performance_rating: 3.9,
                teaching_load: 28,
                years_at_school: 4
            },
            {
                id: 5,
                name: 'Dr. Emily Brown',
                specialization: 'Sciences',
                qualification: 'PhD',
                years_experience: 20,
                experience_level: 'Senior',
                performance_rating: 4.9,
                teaching_load: 22,
                years_at_school: 15
            },
            {
                id: 6,
                name: 'Mr. James Wilson',
                specialization: 'Mathematics',
                qualification: 'HND + PGDE',
                years_experience: 3,
                experience_level: 'Novice',
                performance_rating: 3.5,
                teaching_load: 35,
                years_at_school: 2
            },
            {
                id: 7,
                name: 'Ms. Lisa Garcia',
                specialization: 'General',
                qualification: 'B.Ed',
                years_experience: 10,
                experience_level: 'Experienced',
                performance_rating: 4.3,
                teaching_load: 25,
                years_at_school: 7
            }
        ];
    }

    filterTeachers() {
        this.filteredTeachers = this.teachers.filter(teacher => {
            // Search filter
            if (this.currentFilters.search) {
                const searchTerm = this.currentFilters.search.toLowerCase();
                if (!teacher.name.toLowerCase().includes(searchTerm)) {
                    return false;
                }
            }

            // Specialization filter
            if (this.currentFilters.specialization && teacher.specialization !== this.currentFilters.specialization) {
                return false;
            }

            // Qualification filter
            if (this.currentFilters.qualification && teacher.qualification !== this.currentFilters.qualification) {
                return false;
            }

            // Experience level filter
            if (this.currentFilters.experience && teacher.experience_level !== this.currentFilters.experience) {
                return false;
            }

            return true;
        });

        this.renderTeachers();
        this.updateStats();
    }

    renderTeachers() {
        const tableBody = document.getElementById('teachersTableBody');
        const emptyState = document.getElementById('emptyTeachersState');

        if (!tableBody) return;

        if (this.filteredTeachers.length === 0) {
            tableBody.innerHTML = '';
            if (emptyState) emptyState.style.display = 'flex';
            return;
        }

        if (emptyState) emptyState.style.display = 'none';

        tableBody.innerHTML = this.filteredTeachers.map((teacher, index) => `
            <tr class="${index % 2 === 0 ? 'row-even' : 'row-odd'}">
                <td>
                    <div class="teacher-info">
                        <div class="teacher-avatar">
                            <i class="fas fa-chalkboard-teacher"></i>
                        </div>
                        <div class="teacher-details">
                            <div class="teacher-name">${teacher.name}</div>
                            <div class="teacher-id">ID: ${teacher.teacher_id || teacher.id}</div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="specialization-badge">${teacher.specialization}</span>
                </td>
                <td>
                    <span class="qualification-badge">${teacher.qualification}</span>
                </td>
                <td>
                    <span class="experience-badge experience-${teacher.experience_level.toLowerCase()}">
                        ${teacher.experience_level}
                    </span>
                </td>
                <td>
                    <span class="performance-badge performance-${this.getPerformanceLevel(teacher.performance_rating)}">
                        ${teacher.performance_rating}/5.0
                    </span>
                </td>
                <td>
                    <div class="action-buttons">
                        <button class="btn-action" onclick="this.viewTeacherProfile(${teacher.id})">
                            View Profile
                            <i class="fas fa-arrow-right"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    getPerformanceLevel(rating) {
        if (rating >= 4.5) return 'excellent';
        if (rating >= 4.0) return 'good';
        if (rating >= 3.5) return 'average';
        return 'poor';
    }

    updateStats() {
        const totalCount = document.getElementById('totalTeachersCount');
        const showingCount = document.getElementById('showingTeachersCount');
        const lastUpdated = document.getElementById('lastTeachersUpdated');

        if (totalCount) totalCount.textContent = this.teachers.length;
        if (showingCount) showingCount.textContent = this.filteredTeachers.length;
        if (lastUpdated) lastUpdated.textContent = new Date().toLocaleTimeString();
    }

    showAddTeacherModal() {
        // For now, just show a message. We can implement the modal later
        this.showMessage('Add Teacher feature coming soon!', 'info');
    }

    viewTeacherProfile(teacherId) {
        // Navigate to teacher profile page
        window.location.hash = `#teacher/${teacherId}`;
    }

    showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${type}`;
        messageDiv.textContent = message;
        
        this.container.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }
}
