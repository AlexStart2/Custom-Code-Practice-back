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
}
