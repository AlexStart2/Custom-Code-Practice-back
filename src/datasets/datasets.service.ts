import { Injectable } from '@nestjs/common';
import { StoreRagDto } from './dto/rag_dataset.dto';

@Injectable()
export class DatasetsService {
    async storeRagFiles(data: any[], userId: string) {
        // Logic to handle the file upload and processing
        // This could involve saving files, processing them, etc.
        // For now, we will just return a mock response
        return { success: true, message: 'Arrived here' };
    }
}
