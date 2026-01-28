// UNI TRAINER
// Copyright (c) 2026 [Vagif Hasanov]
// 
// This work was created by [Vagif Hasanov] via - contracting AI tools
// as implementation assistants. All creative direction,
// architecture, design, and business logic are original
// works of [Vagif Hasanov].
// 
// AI was used as a tool under direction, similar to
// how a developer uses a compiler or IDE.

/**
 * CanopyWave API Client
 * Handles all interactions with CanopyWave Cloud API
 * Documentation: https://canopywave.com/docs/account/quick-start
 */

const https = require('https');

class CanopyWaveAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseURL = 'https://cloud-api.canopywave.io/api/v1';
        this.timeout = 30000; // 30 seconds default timeout
    }

    /**
     * Make an API request to CanopyWave
     * @param {string} endpoint - API endpoint (e.g., '/projects')
     * @param {string} method - HTTP method (GET, POST, DELETE, etc.)
     * @param {object} data - Request body data (for POST requests)
     * @param {object} queryParams - Query parameters (e.g., {project: 'proj', region: 'seq'})
     * @returns {Promise<object>} API response (data field extracted)
     */
    async request(endpoint, method = 'GET', data = null, queryParams = {}) {
        return new Promise((resolve, reject) => {
            // Build query string
            const queryString = Object.keys(queryParams)
                .filter(key => queryParams[key] !== null && queryParams[key] !== undefined)
                .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(queryParams[key])}`)
                .join('&');
            
            // Ensure baseURL ends with / and endpoint doesn't start with /
            const base = this.baseURL.endsWith('/') ? this.baseURL : this.baseURL + '/';
            const endpointPath = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
            const fullPath = endpointPath + (queryString ? `?${queryString}` : '');
            const url = new URL(fullPath, base);
            
            const options = {
                hostname: url.hostname,
                port: url.port || 443,
                path: url.pathname + url.search,
                method: method,
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json',
                    'User-Agent': 'UniTrainer/1.0.0'
                },
                timeout: this.timeout
            };

            // Log request details for debugging
            console.log(`[CanopyWave API] ${method} ${options.path}`);
            
            const req = https.request(options, (res) => {
                let responseData = '';

                res.on('data', (chunk) => {
                    responseData += chunk;
                });

                res.on('end', () => {
                    try {
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            const parsed = JSON.parse(responseData);
                            // CanopyWave wraps responses in { "data": ... }
                            // But also check for direct data or error format
                            if (parsed.data !== undefined) {
                                resolve(parsed.data);
                            } else if (parsed.error) {
                                console.error(`[CanopyWave API] Error in successful response:`, parsed.error);
                                reject(new Error(parsed.error));
                            } else {
                                // Direct response (no wrapper)
                                resolve(parsed);
                            }
                        } else {
                            // Error response - log full details
                            console.error(`[CanopyWave API] HTTP ${res.statusCode} Error`);
                            console.error(`[CanopyWave API] Request: ${method} ${options.path}`);
                            console.error(`[CanopyWave API] Response headers:`, JSON.stringify(res.headers, null, 2));
                            console.error(`[CanopyWave API] Response body:`, responseData);
                            
                            try {
                                const error = JSON.parse(responseData);
                                const errorMsg = error.error || error.message || error.detail || error.msg || `API Error (${res.statusCode})`;
                                
                                // Log full error object for debugging
                                console.error(`[CanopyWave API] Parsed error object:`, JSON.stringify(error, null, 2));
                                
                                // Build detailed error message with all available info
                                let detailedError = `CanopyWave API Error (${res.statusCode}): ${errorMsg}`;
                                if (error.detail) {
                                    detailedError += `\nDetails: ${error.detail}`;
                                }
                                if (error.message && error.message !== errorMsg) {
                                    detailedError += `\nMessage: ${error.message}`;
                                }
                                if (error.code) {
                                    detailedError += `\nError Code: ${error.code}`;
                                }
                                if (error.trace || error.stack) {
                                    detailedError += `\nStack: ${error.trace || error.stack}`;
                                }
                                
                                // Provide helpful messages for common errors
                                if (errorMsg.includes('no default payment method')) {
                                    detailedError = 'No payment method configured. Please add a payment method in your CanopyWave dashboard before launching instances.';
                                } else if (res.statusCode === 500) {
                                    detailedError = `CanopyWave server error (500): ${errorMsg}\n\nFull error details:\n${JSON.stringify(error, null, 2)}\n\nThis is a temporary issue on CanopyWave's side. Please try again in a few moments or contact CanopyWave support if the issue persists.`;
                                } else if (res.statusCode === 404) {
                                    detailedError = `Endpoint not found (404): ${options.method} ${options.path}\n\nError: ${errorMsg}\n\nPlease check the API endpoint.`;
                                } else if (res.statusCode === 403) {
                                    detailedError = `Access forbidden (403): ${errorMsg}\n\nFull error: ${JSON.stringify(error, null, 2)}\n\nCheck your API key permissions or account settings.`;
                                } else if (res.statusCode === 401) {
                                    detailedError = `Unauthorized (401): Invalid or expired API key.\n\nError: ${errorMsg}\n\nPlease check your CanopyWave API key.`;
                                }
                                
                                const apiError = new Error(detailedError);
                                // Attach full error details for logging
                                apiError.apiResponse = error;
                                apiError.statusCode = res.statusCode;
                                apiError.endpoint = `${method} ${options.path}`;
                                reject(apiError);
                            } catch (e) {
                                // If response is not JSON, provide more context
                                console.error(`[CanopyWave API] Failed to parse error response as JSON:`, e.message);
                                console.error(`[CanopyWave API] Raw response:`, responseData);
                                const errorMsg = responseData || `HTTP ${res.statusCode} Error`;
                                const apiError = new Error(`CanopyWave API Error (${res.statusCode}): ${errorMsg}`);
                                apiError.statusCode = res.statusCode;
                                apiError.rawResponse = responseData;
                                reject(apiError);
                            }
                        }
                    } catch (e) {
                        console.error('[CanopyWave API] Parse error:', e.message);
                        console.error('[CanopyWave API] Response data:', responseData.substring(0, 500));
                        const parseError = new Error(`Failed to parse CanopyWave API response: ${e.message}\n\nResponse: ${responseData.substring(0, 200)}`);
                        parseError.originalError = e;
                        parseError.rawResponse = responseData;
                        reject(parseError);
                    }
                });
            });

            req.on('error', (error) => {
                console.error(`[CanopyWave API] Request error for ${method} ${options.path}:`, error.message);
                console.error(`[CanopyWave API] Error details:`, error);
                const networkError = new Error(`CanopyWave API network error: ${error.message}\n\nEndpoint: ${method} ${options.path}\n\nThis may indicate a network connectivity issue or CanopyWave API is temporarily unavailable.`);
                networkError.originalError = error;
                networkError.endpoint = `${method} ${options.path}`;
                reject(networkError);
            });

            req.on('timeout', () => {
                console.error(`[CanopyWave API] Request timeout for ${method} ${options.path} (${this.timeout}ms)`);
                req.destroy();
                const timeoutError = new Error(`CanopyWave API request timeout after ${this.timeout}ms\n\nEndpoint: ${method} ${options.path}\n\nThe CanopyWave API may be slow or unavailable. Please try again.`);
                timeoutError.endpoint = `${method} ${options.path}`;
                timeoutError.timeout = this.timeout;
                reject(timeoutError);
            });

            if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
                req.write(JSON.stringify(data));
            }

            req.end();
        });
    }

    /**
     * Validate API key by listing projects
     * @returns {Promise<{valid: boolean, error?: string}>} Validation result
     */
    async validateAPIKey() {
        try {
            const projects = await this.listProjects();
            // If we get a response (even empty array), the key is valid
            if (Array.isArray(projects)) {
                return { valid: true };
            }
            return { valid: false, error: 'Unexpected response format' };
        } catch (error) {
            console.error('API key validation error:', error);
            // Return more detailed error information
            const errorMessage = error.message || 'Unknown error';
            return { valid: false, error: errorMessage };
        }
    }

    /**
     * List projects (required for most API calls)
     * GET /projects
     * @returns {Promise<Array<string>>} List of project names
     */
    async listProjects() {
        return await this.request('/projects', 'GET');
    }

    /**
     * List regions
     * GET /regions?project=<project>
     * @param {string} project - Project name
     * @returns {Promise<Array<string>>} List of region names
     */
    async listRegions(project) {
        return await this.request('/regions', 'GET', null, { project });
    }

    /**
     * List instance types (GPU types/flavors)
     * GET /instance-types?project=<project>
     * @param {string} project - Project name
     * @returns {Promise<Array>} List of available instance types
     */
    async listInstanceTypes(project) {
        return await this.request('/instance-types', 'GET', null, { project });
    }

    /**
     * List images (OS images)
     * GET /images?project=<project>
     * @param {string} project - Project name
     * @returns {Promise<Array<string>>} List of available image names
     */
    async listImages(project) {
        return await this.request('/images', 'GET', null, { project });
    }

    /**
     * Get account balance
     * GET /account/balance
     * @returns {Promise<object>} Balance information
     */
    async getBalance() {
        return await this.request('/account/balance', 'GET');
    }

    /**
     * List instances (VMs)
     * GET /instances?project=<project>&region=<region>
     * @param {string} project - Project name
     * @param {string} region - Region name (optional)
     * @returns {Promise<Array>} List of instances
     */
    async listInstances(project, region = null) {
        return await this.request('/instances', 'GET', null, { project, region });
    }

    /**
     * List running instances only
     * GET /instances/running?project=<project>&region=<region>
     * @param {string} project - Project name
     * @param {string} region - Region name (required for this endpoint)
     * @returns {Promise<Array>} List of running instances
     */
    async listRunningInstances(project, region) {
        return await this.request('/instances/running', 'GET', null, { project, region });
    }

    /**
     * Get instance details
     * GET /instances/<instance-id>?project=<project>&region=<region>
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Instance details
     */
    async getInstance(instanceId, project, region) {
        return await this.request(`/instances/${instanceId}`, 'GET', null, { project, region });
    }

    /**
     * Launch/create a new instance
     * POST /instance-operations/launch
     * @param {object} config - Instance configuration
     * @param {string} config.project - Project name
     * @param {string} config.region - Region name
     * @param {string} config.name - Instance name
     * @param {string} config.flavor - Flavor/instance type (e.g., "H100-4")
     * @param {string} config.image - Image name (e.g., "GPU-Ubuntu.22.04")
     * @param {string} config.password - Password for SSH access
     * @param {string} config.keypair - SSH keypair name (optional)
     * @param {boolean} config.is_monitoring - Enable monitoring (optional)
     * @returns {Promise<object>} Created instance information (with id)
     */
    async launchInstance(config) {
        const payload = {
            project: config.project,
            region: config.region,
            name: config.name,
            flavor: config.flavor,
            image: config.image,
            password: config.password
        };
        
        if (config.keypair) {
            payload.keypair = config.keypair;
        }
        if (config.is_monitoring !== undefined) {
            payload.is_monitoring = config.is_monitoring;
        }
        
        return await this.request('/instance-operations/launch', 'POST', payload);
    }

    /**
     * Terminate an instance
     * POST /instance-operations/terminate
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Termination confirmation
     */
    async terminateInstance(instanceId, project, region) {
        const payload = {
            project: project,
            region: region,
            id: instanceId
        };
        return await this.request('/instance-operations/terminate', 'POST', payload);
    }

    /**
     * Restart an instance
     * POST /instance-operations/restart
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Restart confirmation
     */
    async restartInstance(instanceId, project, region) {
        const payload = {
            project: project,
            region: region,
            id: instanceId
        };
        return await this.request('/instance-operations/restart', 'POST', payload);
    }

    /**
     * List SSH keys
     * GET /ssh-keys?project=<project>&region=<region>
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<Array>} List of SSH keys
     */
    async listSSHKeys(project, region) {
        return await this.request('/ssh-keys', 'GET', null, { project, region });
    }

    /**
     * Check flavor availability
     * GET /flavor-availability?project=<project>&flavor=<flavor>&region=<region>
     * @param {string} project - Project name
     * @param {string} flavor - Flavor name (e.g., "H100-8")
     * @param {string} region - Region name
     * @returns {Promise<object>} Availability information
     */
    async checkFlavorAvailability(project, flavor, region) {
        return await this.request('/flavor-availability', 'GET', null, { project, flavor, region });
    }

    /**
     * List public IP addresses (API #21)
     * GET /ips?project=<project>&region=<region>
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<Array>} List of public IPs
     */
    async listPublicIPs(project, region) {
        return await this.request('/ips', 'GET', null, { project, region });
    }

    /**
     * Allocate a new public IP address (API #22)
     * POST /ips
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Created IP with {id, ip}
     */
    async allocatePublicIP(project, region) {
        const payload = {
            project: project,
            region: region
        };
        return await this.request('/ips', 'POST', payload);
    }

    /**
     * Associate public IP address to instance (API #23)
     * POST /ips/<ipId>/associate
     * @param {string} ipId - IP ID (not the IP address itself)
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Association confirmation with {instanceId, ipId, portId}
     */
    async associatePublicIP(ipId, instanceId, project, region) {
        const payload = {
            region: region,
            project: project,
            instanceId: instanceId
        };
        return await this.request(`/ips/${ipId}/associate`, 'POST', payload);
    }

    /**
     * Disassociate public IP address from instance (API #24)
     * DELETE /ips/<ipId>/disassociate
     * @param {string} ipId - IP ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Disassociation confirmation
     */
    async disassociatePublicIP(ipId, project, region) {
        return await this.request(`/ips/${ipId}/disassociate`, 'DELETE', null, { project, region });
    }

    /**
     * Release/delete public IP address (API #25)
     * DELETE /ips/<id>
     * @param {string} ipId - IP ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Release confirmation
     */
    async releasePublicIP(ipId, project, region) {
        return await this.request(`/ips/${ipId}`, 'DELETE', null, { project, region });
    }

    // Legacy aliases for backward compatibility
    async listFloatingIPs(project, region) {
        return await this.listPublicIPs(project, region);
    }

    async createFloatingIP(project, region) {
        return await this.allocatePublicIP(project, region);
    }

    async associateFloatingIP(ipId, instanceId, project, region) {
        return await this.associatePublicIP(ipId, instanceId, project, region);
    }

    async disassociateFloatingIP(ipId, project, region) {
        return await this.disassociatePublicIP(ipId, project, region);
    }

    // Convenience methods for training workflow (aliases for compatibility)

    /**
     * List available GPU types/resources (wraps listInstanceTypes)
     * @param {string} project - Project name
     * @returns {Promise<Array>} Available GPU instance types
     */
    async listAvailableGPUs(project) {
        return await this.listInstanceTypes(project);
    }

    /**
     * Get instance status (wraps getInstance)
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Instance status
     */
    async getJobStatus(instanceId, project, region) {
        return await this.getInstance(instanceId, project, region);
    }

    /**
     * List all jobs/instances (wraps listInstances)
     * @param {string} project - Project name
     * @param {string} region - Region name (optional)
     * @returns {Promise<Array>} List of instances
     */
    async listJobs(project, region = null) {
        return await this.listInstances(project, region);
    }

    /**
     * Cancel/stop a training job (wraps terminateInstance)
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Termination confirmation
     */
    async cancelJob(instanceId, project, region) {
        return await this.terminateInstance(instanceId, project, region);
    }

    /**
     * Create/launch instance for training (wraps launchInstance)
     * @param {object} config - Training instance configuration
     * @returns {Promise<object>} Instance information
     */
    async createInstance(config) {
        return await this.launchInstance(config);
    }

    /**
     * Delete/terminate instance (wraps terminateInstance)
     * @param {string} instanceId - Instance ID
     * @param {string} project - Project name
     * @param {string} region - Region name
     * @returns {Promise<object>} Termination confirmation
     */
    async deleteInstance(instanceId, project, region) {
        return await this.terminateInstance(instanceId, project, region);
    }
}

module.exports = CanopyWaveAPI;
