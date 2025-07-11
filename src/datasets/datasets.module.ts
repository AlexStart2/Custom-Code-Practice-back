import { Module } from '@nestjs/common';
import { DatasetsController } from './datasets.controller';
import { mongo } from 'mongoose';
import { MongooseModule } from '@nestjs/mongoose';
import { DatasetsService } from './datasets.service';

@Module({
  controllers: [DatasetsController],
  providers: [DatasetsService]
})
export class DatasetsModule {}
