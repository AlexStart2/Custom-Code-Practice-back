import { Injectable, HttpException } from '@nestjs/common';
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

    async getAllDatasets(UserId: string) {
        // Search for datasets in the database using the UserId

        if (!Types.ObjectId.isValid(UserId)) {
            throw new HttpException('Invalid User ID format', 400);
        }

        const datasets = await this.datasetModel.find({ owner: UserId }).exec();

        if (!datasets) {
            throw new HttpException('No datasets found for this user', 404);
        }

        if (!datasets.some(dataset => dataset.files.length > 0)) {
            throw new HttpException('No files found for these datasets', 404);
        }

        return datasets;
    }


    async deleteDataset(UserId: string, datasetId: string) {

        // Validate the datasetId format
        if (!Types.ObjectId.isValid(datasetId)) {
            throw new HttpException('Invalid dataset ID format', 400);
        }

        // Validate the UserId format
        if (!Types.ObjectId.isValid(UserId)) {
            throw new HttpException('Invalid User ID format', 400);
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: UserId }).exec();

        if (!dataset) {
            throw new HttpException('Dataset not found or does not belong to the user', 404);
        }

        // Delete the dataset
        try{
            await this.datasetModel.deleteOne({ _id: datasetId }).exec();
        }catch (error) {
            throw new HttpException('Error deleting dataset', 500);
        }
        
        return { message: 'Dataset deleted successfully' };
    }


    async getDatasetById(UserId: string, datasetId: string) {


        if (!Types.ObjectId.isValid(datasetId)) {
            throw new HttpException('Invalid dataset ID format', 400);
        }
        if (!Types.ObjectId.isValid(UserId)) {
            throw new HttpException('Invalid User ID format', 400);
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: UserId }).exec();
        if (!dataset) {
            throw new HttpException('Dataset not found or does not belong to the user', 404);
        }

        let files: ProcessedFileDocument[] = [];

        for (const fileId of dataset.files) {
            if (!Types.ObjectId.isValid(fileId)) {
                throw new HttpException(`Invalid file ID format: ${fileId}`, 400);
            }
            const file = await this.processedFileModel.findOne({ _id: fileId }).exec();
            if (!file) {
                throw new HttpException(`File not found: ${fileId}`, 404);
            }
            files.push(file);
        }

        return { dataset, files };
    }


    async getJobByUserId(UserId: string) {
        // Find the job by ID and ensure it belongs to the user

        if (!Types.ObjectId.isValid(UserId)) {
            throw new Error(`Invalid user ID format: ${UserId}`);
        }

        const job = await this.jobModel.find({ owner: UserId }).exec();

        if (!job) {
            throw new HttpException('Job not found or does not belong to the user', 404);
        }


        let processingFiles: ProcessingFileDocument[] = [];

        for (const j of job) {
            const files = await this.processingFileModel.find({ job_id: j._id }).exec();
            processingFiles.push(...files);
        }

        if (!processingFiles) {
            throw new HttpException('No processing files found for this job', 404);
        }

        return { job, processingFiles };
    }


    async updateDatasetName(UserId: string, datasetId: string, name: string) {
        // Validate the datasetId format
        if (!Types.ObjectId.isValid(datasetId)) {
            throw new HttpException('Invalid dataset ID format', 400);
        }

        // Validate the UserId format
        if (!Types.ObjectId.isValid(UserId)) {
            throw new HttpException('Invalid User ID format', 400);
        }

        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: UserId }).exec();

        if (!dataset) {
            throw new HttpException('Dataset not found or does not belong to the user', 404);
        }

        // Update the dataset name
        dataset.name = name;
        await dataset.save();

        return { message: 'Dataset name updated successfully' };
    }
}
