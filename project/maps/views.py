from django.shortcuts import render

# Create your views here.
def default_map(request):
        # TODO: move this token to Django settings from an environment variable
    # found in the Mapbox account settings and getting started instructions
    # see https://www.mapbox.com/account/ under the "Access tokens" section
    mapbox_access_token = 'pk.eyJ1Ijoic2hldHR5a2FyYW41NyIsImEiOiJja3VwNHNuaGwyM242MzNvNnNlMm1obTdyIn0.1zYRiUsXVCn4YMJ94-Ng-Q'
    return render(request, 'default.html', 
                  { 'mapbox_access_token': mapbox_access_token })
#    return render(request, 'default.html', {})
