import { Injectable, NotFoundException, BadRequestException, InternalServerErrorException, ForbiddenException } from '@nestjs/common';
import { Dataset, Job, ProcessingFile, ProcessedFile } from './schemas/dataset.schema';
import { Model, Types } from 'mongoose';
import { InjectModel } from '@nestjs/mongoose';
import { DatasetsDocument, JobDocument, ProcessingFileDocument, ProcessedFileDocument } from './schemas/dataset.schema';


@Injectable()
export class DatasetsService {
    constructor(
        @InjectModel(Dataset.name) private datasetModel: Model<DatasetsDocument>,
        @InjectModel(Job.name) private jobModel: Model<JobDocument>,
        @InjectModel(ProcessingFile.name) private processingFileModel: Model<ProcessingFileDocument>,
        @InjectModel(ProcessedFile.name) private processedFileModel: Model<ProcessedFileDocument>,
    ) {}

    async getAllDatasets(userId: string) {
        // Search for datasets in the database using the userId

        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException('Invalid User ID format');
        }

        const datasets = await this.datasetModel.find({ owner: userId }).exec();

        if (!datasets || datasets.length === 0) {
            return []; // Return empty array instead of throwing error
        }

        return datasets;
    }


    async deleteDataset(userId: string, datasetId: string) {

        // Validate the datasetId format
        if (!Types.ObjectId.isValid(datasetId)) {
            throw new BadRequestException('Invalid dataset ID format');
        }

        // Validate the UserId format
        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException('Invalid User ID format');
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: userId }).exec();

        if (!dataset) {
            throw new NotFoundException('Dataset not found or does not belong to the user');
        }

        // Delete the dataset
        try {
            await this.datasetModel.deleteOne({ _id: datasetId }).exec();
        } catch (error) {
            throw new InternalServerErrorException('Error deleting dataset');
        }
        
        return { message: 'Dataset deleted successfully' };
    }


    async getDatasetById(userId: string, datasetId: string) {

        if (!Types.ObjectId.isValid(datasetId)) {
            throw new BadRequestException('Invalid dataset ID format');
        }
        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException('Invalid User ID format');
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: userId }).exec();
        if (!dataset) {
            throw new NotFoundException('Dataset not found or does not belong to the user');
        }

        const files: ProcessedFileDocument[] = [];

        for (const fileId of dataset.files) {
            if (!Types.ObjectId.isValid(fileId)) {
                throw new BadRequestException(`Invalid file ID format: ${fileId}`);
            }
            const file = await this.processedFileModel.findOne({ _id: fileId }).exec();
            if (!file) {
                throw new NotFoundException(`File not found: ${fileId}`);
            }
            files.push(file);
        }

        return { dataset, files };
    }


    async getJobByUserId(userId: string) {
        // Find the job by ID and ensure it belongs to the user

        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException(`Invalid user ID format: ${userId}`);
        }

        const jobs = await this.jobModel.find({ owner: userId }).exec();

        if (!jobs || jobs.length === 0) {
            return { jobs: [], processingFiles: [] };
        }

        const processingFiles: ProcessingFileDocument[] = [];

        for (const job of jobs) {
            const files = await this.processingFileModel.find({ job_id: job._id }).exec();
            processingFiles.push(...files);
        }

        return { jobs, processingFiles };
    }


    async updateDatasetName(userId: string, datasetId: string, name: string) {
        // Validate the datasetId format
        if (!Types.ObjectId.isValid(datasetId)) {
            throw new BadRequestException('Invalid dataset ID format');
        }

        // Validate the UserId format
        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException('Invalid User ID format');
        }

        // Validate the name
        if (!name || name.trim().length === 0) {
            throw new BadRequestException('Name cannot be empty');
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: userId }).exec();

        if (!dataset) {
            throw new NotFoundException('Dataset not found or does not belong to the user');
        }

        // Update the dataset name
        dataset.name = name.trim();
        await dataset.save();

        return { message: 'Dataset name updated successfully', dataset: { id: dataset._id, name: dataset.name } };
    }


    async deleteChunk(userId: string, datasetId: string, fileId: string, idx: number) {
        // Validate the datasetId format
        if (!Types.ObjectId.isValid(datasetId)) {
            throw new BadRequestException('Invalid dataset ID format');
        }

        // Validate the UserId format
        if (!Types.ObjectId.isValid(userId)) {
            throw new BadRequestException('Invalid User ID format');
        }

        // Validate the fileId format
        if (!Types.ObjectId.isValid(fileId)) {
            throw new BadRequestException('Invalid file ID format');
        }

        // Validate chunk index
        if (idx < 0 || !Number.isInteger(idx)) {
            throw new BadRequestException('Invalid chunk index');
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: userId }).exec();

        if (!dataset) {
            throw new NotFoundException('Dataset not found or does not belong to the user');
        }

        // Check if the file exists in the dataset
        const fileExists = dataset.files.some(file => file.toString() === fileId);
        if (!fileExists) {
            throw new NotFoundException('File not found in the dataset');
        }

        // Delete the chunk from the processed file
        const processedFile = await this.processedFileModel.findOne({ _id: fileId }).exec();
        if (!processedFile || !processedFile.results || idx >= processedFile.results.length) {
            throw new NotFoundException('Chunk not found');
        }

        processedFile.results.splice(idx, 1);
        await processedFile.save();

        return { 
            message: 'Chunk deleted successfully',
            remainingChunks: processedFile.results.length
        };
    }
}
