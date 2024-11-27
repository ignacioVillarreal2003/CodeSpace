import { Component } from '@angular/core';
import {NgForOf, NgStyle} from "@angular/common";
import {Topic} from '../../content/topic';
import {DataService} from '../../services/data.service';
import {ActivatedRoute, Router} from '@angular/router';
import {Unit} from '../../content/unit';

@Component({
  selector: 'app-units-overview',
  standalone: true,
  imports: [
    NgForOf,
    NgStyle
  ],
  templateUrl: './units-overview.component.html',
  styleUrl: './units-overview.component.css'
})
export class UnitsOverviewComponent {
  units: Unit[] = []

  constructor(private dataService: DataService, private route: ActivatedRoute, private router: Router) {}

  ngOnInit(): void {
    const topicId: string | null = this.route.snapshot.paramMap.get('topicId');
    if (topicId != null) {
      this.units = this.dataService.getUnits(topicId);
    }
  }

  openUnit(unitId: string): void {
    this.router.navigate([`unitContent/${unitId}`])
  }
}
