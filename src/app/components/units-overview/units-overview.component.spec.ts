import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UnitsOverviewComponent } from './units-overview.component';

describe('UnitsOverviewComponent', () => {
  let component: UnitsOverviewComponent;
  let fixture: ComponentFixture<UnitsOverviewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UnitsOverviewComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UnitsOverviewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
