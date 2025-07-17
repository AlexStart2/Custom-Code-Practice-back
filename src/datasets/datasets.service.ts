import { Injectable } from '@nestjs/common';
import { Dataset } from './schemas/dataset.schema';
import { Model } from 'mongoose';
import { InjectModel } from '@nestjs/mongoose';
import { DatasetsDocument } from './schemas/dataset.schema';


@Injectable()
export class DatasetsService {
    constructor(@InjectModel(Dataset.name) private datasetModel: Model<DatasetsDocument>) {}

    async getAllDatasets(UserId: string) {
        // Search for datasets in the database using the UserId
        const datasets = await this.datasetModel.find({ owner: UserId }).exec();

        return datasets;
    }


    async deleteDataset(UserId: string, datasetId: string) {
        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: UserId }).exec();

        if (!dataset) {
            throw new Error('Dataset not found or does not belong to the user');
        }

        // Delete the dataset
        try{
            await this.datasetModel.deleteOne({ _id: datasetId }).exec();
        }catch (error) {
            throw new Error('Error deleting dataset');
        }
        
        return { message: 'Dataset deleted successfully' };
    }


    async getDatasetById(UserId: string, datasetId: string) {
        // Find the dataset by ID and ensure it belongs to the user
        const dataset = await this.datasetModel.findOne({ _id: datasetId, owner: UserId }).exec();

        if (!dataset) {
            throw new Error('Dataset not found or does not belong to the user');
        }

        return dataset;
    }
}
