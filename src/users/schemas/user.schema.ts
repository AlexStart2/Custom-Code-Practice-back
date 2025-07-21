import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import mongoose, { Document } from 'mongoose';

export type UserDocument = User & Document;

@Schema({collection: 'users'})
export class User {
    // @Prop({ type: mongoose.Types.ObjectId })
    _id: string;

    @Prop({ required: true, unique: true })
    email: string;

    @Prop({ required: true })
    password: string;

    @Prop({ required: true })
    name: string;
}

export const UserSchema = SchemaFactory.createForClass(User);


UserSchema.virtual('id').get(function (this: UserDocument) {
    return this._id.toString();
});

UserSchema.set('toJSON', {
    virtuals: true,
});
UserSchema.set('toObject', {
    virtuals: true,
});


class messages {
    @Prop({ type: String, required: true })
    prompt: string;

    @Prop({ type: String, required: false})
    answer: string;

    @Prop({ type: Date, default: Date.now })
    createdAt: Date;
}

export type HistoryDocument = History & Document;


@Schema({collection: 'history'})
export class History {
    @Prop({ type: mongoose.Types.ObjectId, required: true })
    owner: mongoose.Types.ObjectId;

    @Prop({ type: mongoose.Types.ObjectId, required: false })
    dataset: mongoose.Types.ObjectId;

    @Prop({ type: [messages], required: true })
    messages: messages[];

    @Prop({ type: String, required: false })
    model: string;

    @Prop({ type: Date, default: Date.now })
    createdAt: Date;
}

export const HistorySchema = SchemaFactory.createForClass(History);