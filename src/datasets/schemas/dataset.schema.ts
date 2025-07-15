
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import mongoose, { Document } from 'mongoose';

export type DatasetsDocument = Dataset & Document;

@Schema({collection: 'datasets_rag'})
export class Dataset{

    _id: string;

    @Prop({required: true})
    owner: string;

    @Prop({required: true})
    name: string;

    @Prop({required: true})
    chunks: any[];

    @Prop({required: true})
    createdAt: Date;
}

export const DatasetSchema = SchemaFactory.createForClass(Dataset);