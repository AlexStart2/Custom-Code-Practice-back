import { Module } from '@nestjs/common';
import { DatasetsController } from './datasets.controller';
import { MongooseModule } from '@nestjs/mongoose';
import { DatasetsService } from './datasets.service';
import { Dataset, DatasetSchema, Job, JobSchema, ProcessingFile, 
  ProcessingFileSchema, ProcessedFile, ProcessedFileSchema } from './schemas/dataset.schema';


@Module({
  imports: [
    MongooseModule.forFeature([
      { name: Dataset.name, schema: DatasetSchema },
      { name: Job.name, schema: JobSchema },
      { name: ProcessingFile.name, schema: ProcessingFileSchema },
      { name: ProcessedFile.name, schema: ProcessedFileSchema },
    ])
  ],
  controllers: [DatasetsController],
  providers: [DatasetsService],
  exports: [DatasetsService]
})
export class DatasetsModule {}