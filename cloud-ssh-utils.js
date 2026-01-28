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
 * Cloud SSH Utilities
 * Handles SSH connections, file transfers, and remote command execution for cloud training instances
 */

const { Client } = require('ssh2');
const fs = require('fs');
const path = require('path');

class CloudSSHUtils {
    constructor(host, username, password, port = 22) {
        this.host = host;
        this.username = username;
        this.password = password;
        this.port = port;
        this.conn = null;
    }

    /**
     * Connect to the remote instance
     * @returns {Promise<void>}
     */
    async connect() {
        return new Promise((resolve, reject) => {
            this.conn = new Client();
            
            this.conn.on('ready', () => {
                console.log('[SSH] Connection established');
                resolve();
            });

            this.conn.on('error', (err) => {
                console.error('[SSH] Connection error:', err);
                reject(err);
            });

            this.conn.connect({
                host: this.host,
                port: this.port,
                username: this.username,
                password: this.password,
                readyTimeout: 30000
            });
        });
    }

    /**
     * Execute a command on the remote instance
     * @param {string} command - Command to execute
     * @param {number} timeout - Timeout in milliseconds (default 5 minutes)
     * @returns {Promise<{stdout: string, stderr: string, code: number}>}
     */
    async executeCommand(command, timeout = 300000) {
        if (!this.conn) {
            throw new Error('SSH connection not established. Call connect() first.');
        }

        return new Promise((resolve, reject) => {
            let stdout = '';
            let stderr = '';
            let timeoutHandle = null;

            this.conn.exec(command, (err, stream) => {
                if (err) {
                    clearTimeout(timeoutHandle);
                    reject(err);
                    return;
                }

                // Set timeout
                timeoutHandle = setTimeout(() => {
                    stream.close();
                    reject(new Error(`Command timeout after ${timeout}ms: ${command}`));
                }, timeout);

                stream.on('close', (code, signal) => {
                    clearTimeout(timeoutHandle);
                    resolve({
                        stdout: stdout.trim(),
                        stderr: stderr.trim(),
                        code: code,
                        signal: signal
                    });
                });

                stream.on('data', (data) => {
                    stdout += data.toString();
                });

                stream.stderr.on('data', (data) => {
                    stderr += data.toString();
                });
            });
        });
    }

    /**
     * Upload a file to the remote instance
     * @param {string} localPath - Local file path
     * @param {string} remotePath - Remote file path
     * @returns {Promise<void>}
     */
    async uploadFile(localPath, remotePath) {
        if (!this.conn) {
            throw new Error('SSH connection not established. Call connect() first.');
        }

        return new Promise((resolve, reject) => {
            this.conn.sftp((err, sftp) => {
                if (err) {
                    reject(err);
                    return;
                }

                // Create remote directory if needed (expand ~ to home directory)
                let remoteDir = remotePath.replace(/^~/, '/home/' + this.username);
                remoteDir = remoteDir.substring(0, remoteDir.lastIndexOf('/'));
                // Use proper shell expansion for mkdir
                const mkdirPath = remotePath.replace(/^~/, '$HOME').substring(0, remotePath.lastIndexOf('/'));
                
                this.executeCommand(`mkdir -p "${mkdirPath}"`)
                    .then(() => {
                        // Expand ~ in remote path for SFTP
                        const expandedRemotePath = remotePath.replace(/^~/, '/home/' + this.username);
                        sftp.fastPut(localPath, expandedRemotePath, (err) => {
                            if (err) {
                                reject(new Error(`SFTP upload failed: ${err.message}. Local: ${localPath}, Remote: ${expandedRemotePath}`));
                            } else {
                                console.log(`[SSH] Uploaded ${localPath} to ${expandedRemotePath}`);
                                resolve();
                            }
                        });
                    })
                    .catch(reject);
            });
        });
    }

    /**
     * Upload a directory recursively to the remote instance
     * @param {string} localDir - Local directory path
     * @param {string} remoteDir - Remote directory path
     * @returns {Promise<void>}
     */
    async uploadDirectory(localDir, remoteDir) {
        if (!this.conn) {
            throw new Error('SSH connection not established. Call connect() first.');
        }

        return new Promise((resolve, reject) => {
            this.conn.sftp((err, sftp) => {
                if (err) {
                    reject(err);
                    return;
                }

                // Create remote directory
                this.executeCommand(`mkdir -p "${remoteDir}"`)
                    .then(() => {
                        // Upload files recursively
                        this._uploadDirectoryRecursive(sftp, localDir, remoteDir)
                            .then(resolve)
                            .catch(reject);
                    })
                    .catch(reject);
            });
        });
    }

    /**
     * Helper method to recursively upload directory
     * @private
     */
    async _uploadDirectoryRecursive(sftp, localDir, remoteDir) {
        const files = fs.readdirSync(localDir);

        for (const file of files) {
            const localPath = path.join(localDir, file);
            const remotePath = path.join(remoteDir, file).replace(/\\/g, '/');
            const stat = fs.statSync(localPath);

            if (stat.isDirectory()) {
                await this.executeCommand(`mkdir -p "${remotePath}"`);
                await this._uploadDirectoryRecursive(sftp, localPath, remotePath);
            } else {
                await new Promise((resolve, reject) => {
                    sftp.fastPut(localPath, remotePath, (err) => {
                        if (err) reject(err);
                        else resolve();
                    });
                });
            }
        }
    }

    /**
     * Download a file from the remote instance
     * @param {string} remotePath - Remote file path
     * @param {string} localPath - Local file path
     * @returns {Promise<void>}
     */
    async downloadFile(remotePath, localPath) {
        if (!this.conn) {
            throw new Error('SSH connection not established. Call connect() first.');
        }

        // SFTP fastGet doesn't expand ~, so ensure we have an absolute path
        // (Caller should normalize, but defensive check here)
        let normalizedRemotePath = remotePath;
        if (remotePath.startsWith('~/')) {
            normalizedRemotePath = remotePath.replace('~/', `/home/${this.username}/`);
            console.warn(`[SSH] Warning: downloadFile received ~ path, normalized to: ${normalizedRemotePath}`);
        }

        return new Promise((resolve, reject) => {
            this.conn.sftp((err, sftp) => {
                if (err) {
                    reject(err);
                    return;
                }

                // Create local directory if needed
                const localDir = path.dirname(localPath);
                if (!fs.existsSync(localDir)) {
                    fs.mkdirSync(localDir, { recursive: true });
                }

                sftp.fastGet(normalizedRemotePath, localPath, (err) => {
                    if (err) {
                        reject(new Error(`SFTP download failed: ${err.message}. Remote: ${normalizedRemotePath}, Local: ${localPath}`));
                    } else {
                        console.log(`[SSH] Downloaded ${normalizedRemotePath} to ${localPath}`);
                        resolve();
                    }
                });
            });
        });
    }

    /**
     * Close the SSH connection
     */
    close() {
        if (this.conn) {
            this.conn.end();
            this.conn = null;
        }
    }
}

module.exports = CloudSSHUtils;
