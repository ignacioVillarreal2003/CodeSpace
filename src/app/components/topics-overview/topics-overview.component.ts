import { Component } from '@angular/core';
import {NgForOf, NgStyle} from "@angular/common";
import {Topic} from '../../content/topic';
import {DataService} from '../../services/data.service';
import {ActivatedRoute, Router} from '@angular/router';

@Component({
  selector: 'app-topics-overview',
  standalone: true,
  imports: [
    NgForOf,
    NgStyle
  ],
  templateUrl: './topics-overview.component.html',
  styleUrl: './topics-overview.component.css'
})
export class TopicsOverviewComponent {
  topics: Topic[] = []

  constructor(private dataService: DataService, private route: ActivatedRoute, private router: Router) {}

  ngOnInit(): void {
    const categoryId: string | null = this.route.snapshot.paramMap.get('categoryId');
    if (categoryId != null) {
      this.topics = this.dataService.getTopics(categoryId);
    }
  }

  openTopic(topicId: string): void {
    this.router.navigate([`unitsOverview/${topicId}`])
  }
}
