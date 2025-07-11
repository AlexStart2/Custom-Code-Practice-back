import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import mongoose, { Document } from 'mongoose';

export type UserDocument = User & Document;

@Schema()
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