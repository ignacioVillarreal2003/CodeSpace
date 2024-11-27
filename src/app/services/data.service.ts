import { Injectable } from '@angular/core';
import { topics } from '../content/topic';
import { Topic } from '../content/topic';
import { categories } from '../content/category';
import { Category } from '../content/category';
import {Unit} from '../content/unit';
import {units} from '../content/unit';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  constructor() { }

  getCategories(): Category[] {
    return categories;
  }

  getTopics(categoryId: string): Topic[] {
    return topics.filter((topic: Topic): boolean => topic.categoryId == categoryId);
  }

  getUnits(topicId: string): Unit[] {
    return units.filter((unit: Unit): boolean => unit.topicId == topicId);
  }

  getUnit(unitId: string): Unit | undefined {
    return units.find((unit: Unit): boolean => unit.id == unitId);
  }
}
