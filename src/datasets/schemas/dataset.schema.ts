
import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import mongoose, { Document } from 'mongoose';

export type DatasetsDocument = Dataset & Document;

@Schema({collection: 'datasets_rag'})
export class Dataset{

    @Prop({type: mongoose.Schema.Types.ObjectId, auto: true})
    _id: string;

    @Prop({required: true})
    owner: string;

    @Prop({required: true})
    name: string;

    @Prop({required: true})
    files: string[];

    @Prop({required: true})
    createdAt: Date;
}

export const DatasetSchema = SchemaFactory.createForClass(Dataset);

export type JobDocument = Job & Document;

@Schema({collection: 'jobs_rag'})
export class Job {

    @Prop({type: mongoose.Schema.Types.ObjectId, auto: true})
    _id: string;

    @Prop({required: true})
    owner: string;

    @Prop({required: true})
    dataset_name: string;

    @Prop({required: true})
    status: string;

    @Prop({required: true})
    createdAt: Date;

    @Prop({required: false})
    finishedAt: Date;

    @Prop({required: false})
    error: string;
}

export const JobSchema = SchemaFactory.createForClass(Job);


export type ProcessingFileDocument = ProcessingFile & Document;

@Schema({collection: 'processing_files_rag'})
export class ProcessingFile {

    @Prop({type: mongoose.Schema.Types.ObjectId, auto: true})
    _id: string;

    @Prop({required: true})
    job_id: string;

    @Prop({required: true})
    file_name: string;

    @Prop({required: true})
    status: string;

    @Prop({required: true})
    createdAt: Date;

    @Prop({required: false})
    finishedAt: Date;

    @Prop({required: false})
    error: string;
}

export const ProcessingFileSchema = SchemaFactory.createForClass(ProcessingFile);


export type ProcessedFileDocument = ProcessedFile & Document;

@Schema({collection: 'processed_files'})
export class ProcessedFile {

    @Prop({type: mongoose.Schema.Types.ObjectId, auto: true})
    _id: string;

    @Prop({required: true})
    file_name: string;

    @Prop({required: true})
    results: {
        text: string;
        embedding: number[];
    }[];
}

export const ProcessedFileSchema = SchemaFactory.createForClass(ProcessedFile);