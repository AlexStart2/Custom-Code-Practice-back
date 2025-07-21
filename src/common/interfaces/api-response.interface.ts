export interface ApiResponse<T = any> {
    success: boolean;
    data?: T;
    message?: string;
    timestamp: Date;
}

export function createResponse<T>(data: T, message?: string): ApiResponse<T> {
    return {
        success: true,
        data,
        message: message || 'Operation completed successfully',
        timestamp: new Date()
    };
}

export function createErrorResponse(message: string): ApiResponse<null> {
    return {
        success: false,
        data: null,
        message,
        timestamp: new Date()
    };
}
